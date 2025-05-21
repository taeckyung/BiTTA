import os

import math
import pandas as pd
import torch.nn
from torch import optim
from torch.nn import KLDivLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import conf
from utils import mecta
from utils.sam_optimizer import SAM
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
import copy
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class BITTA_PARALLEL(DNN):
    """
    Code is partially inspired from:
    - https://github.com/taeckyung/SoTTA/blob/main/learner/sotta.py
    - https://github.com/tmllab/2023_NeurIPS_FlatMatch/blob/main/trainer.py
    """

    def __init__(self, *args, **kwargs):
        self.atta_src_net = None
        super(BITTA_PARALLEL, self).__init__(*args, **kwargs)
        # self.src_net = copy.deepcopy(self.net)
        # self.src_net_state = copy.deepcopy(self.net.state_dict())
        self.loss_fn = softmax_cross_entropy()
        assert (conf.args.memory_type == "ActivePriorityFIFO")

    def init_learner(self):
        for param in self.net.parameters():  # turn on requires_grad for all
            param.requires_grad = True

        for name, module in self.net.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'fc' in name:
                for param in module.parameters():
                    param.requires_grad = True
                continue

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats:  # use learn stats
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        if conf.args.enable_mecta:
            self.net = mecta.prepare_model(self.net).to(device)

        optimizer = torch.optim.SGD(
                        self.net.parameters(),
                        conf.args.opt['learning_rate'],
                        momentum=conf.args.opt['momentum'],
                        weight_decay=conf.args.opt['weight_decay'],
                        nesterov=True)

        self.src_net_state = copy.deepcopy(self.net.state_dict())
        self.cnt_correct_after_frze = 0

        # if conf.args.model == "resnet50_domainnet":
        #     self.dropout_nets = [self.net] + [
        #         self.deepcopy_resnet50_domainnet(self.net).to(torch.device("cuda:{:d}".format((conf.args.gpu_idx + i) % 8)))
        #         for i in range(1, conf.args.n_dropouts)
        #     ]
        # else:
        #     self.dropout_nets = [self.net] + [
        #         copy.deepcopy(self.net).to(torch.device("cuda:{:d}".format((conf.args.gpu_idx + i) % 8)))
        #         for i in range(1, conf.args.n_dropouts)
        #     ]


        return optimizer

    def test_time_adaptation(self):
        feats, labels, feedbacks = [], [], []
        len_c, len_w, len_u = 0, 0, 0

        accumulation_steps = 4

        # Unlabeled data
        u_feats, u_labels, _, _ = self.mem.get_u_memory()
        if len(u_feats) != 0:
            feats.append(torch.stack(u_feats).to(device))
            labels.append(torch.tensor(u_labels).to(device))
            feedbacks.append(torch.zeros(len(u_labels)).to(device))
            len_u = len(u_labels)

        # Correct data
        correct_feats, correct_labels, _, _ = self.mem.get_correct_memory()
        self.json_active['num_correct_mem'] += [len(correct_labels)]
        if len(correct_feats) != 0:
            feats.append(torch.stack(correct_feats).to(device))
            labels.append(torch.tensor(correct_labels).to(device))
            feedbacks.append(torch.ones(len(correct_labels)).to(device))
            len_c = len(correct_labels)

        # Wrong data
        wrong_feats, wrong_labels, wrong_gt_labels, _ = self.mem.get_wrong_memory()
        self.json_active['num_wrong_mem'] += [len(wrong_labels)]
        if len(wrong_feats) != 0:
            feats.append(torch.stack(wrong_feats).to(device))
            labels.append(torch.tensor(wrong_labels).to(device))
            feedbacks.append(- torch.ones(len(wrong_labels)).to(device))
            len_w = len(wrong_labels)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        feedbacks = torch.cat(feedbacks, dim=0)
        dataset = torch.utils.data.TensorDataset(feats, labels, feedbacks)
        dataloader = DataLoader(dataset, batch_size=int(conf.args.update_every_x / accumulation_steps), shuffle=True, drop_last=False, pin_memory=False)

        self.net.train()
        self.disable_running_stats()
        self.optimizer.zero_grad()

        epoch = conf.args.epoch
        for e in range(epoch):
            for i, (feats_, labels_, binary_feedback_) in enumerate(dataloader):
                # For optimization
                mcd_softmaxs, mcd_mean_softmax, _ = self.dropout_inference(feats_, conf.args.n_dropouts, dropout=conf.args.dropout_rate)
                mc_dropout_softmax_aggregated = mcd_mean_softmax

                with torch.no_grad():
                    outputs = self.net(feats_)
                    outputs_softmax = outputs.softmax(dim=1)
                    preds = outputs.argmax(dim=1)

                correct_loss, wrong_loss, unlabeled_loss = self.get_loss(mcd_mean_softmax, outputs_softmax, labels_, binary_feedback_, len_c, len_w, len_u)

                loss = conf.args.w_final_loss_unlabeled * unlabeled_loss\
                       + (conf.args.w_final_loss_correct * correct_loss + conf.args.w_final_loss_wrong * wrong_loss)  # averaging on correct/incorrect

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # if conf.args.wandb:
                #     u_idx = binary_feedback_ == 0
                #     u_labels_ = labels_[u_idx]
                #     pred_agreement = mcd_mean_softmax.argmax(dim=1)[u_idx] == preds[u_idx]
                #     orig_conf = outputs_softmax[torch.arange(len(preds)), preds][u_idx]
                #     mcd_conf = mc_dropout_softmax_aggregated[torch.arange(len(preds)), preds][u_idx]
                #
                #     import wandb
                #     wandb.log({
                #         'num_batch_adapt': self.num_batch_adapt + float(e) / conf.args.epoch,
                #         'loss_correct': correct_loss.item(),
                #         'loss_wrong': wrong_loss.item(),
                #         'loss_unlabeled': unlabeled_loss.item(),
                #         'loss_total': loss.item(),
                #         'agreement_ratio': torch.sum(pred_agreement) / torch.sum(u_idx),
                #         'pred_agreement_agree_acc': (preds[u_idx][pred_agreement] == u_labels_[pred_agreement]).sum() / torch.sum(pred_agreement),
                #         'pred_agreement_disagree_acc': (preds[u_idx][~pred_agreement] == u_labels_[~pred_agreement]).sum() / torch.sum(~pred_agreement),
                #         'correct_avg_pred_agreement': pred_agreement[preds[u_idx] == u_labels_].sum() / torch.sum(preds[u_idx] == u_labels_),
                #         'incorrect_avg_pred_agreement': pred_agreement[preds[u_idx] != u_labels_].sum() / torch.sum(preds[u_idx] != u_labels_),
                #         'avg_conf_pred_agreement': orig_conf[pred_agreement].mean(),
                #         'avg_conf_pred_disagreement': orig_conf[~pred_agreement].mean(),
                #         # 'conf_threshold_0.99_ratio': torch.sum(orig_conf > 0.99) / len(u_labels_),
                #         # 'conf_threshold_0.99_high_acc': (preds[u_idx][orig_conf > 0.99] == u_labels_[orig_conf > 0.99]).sum() / torch.sum(orig_conf > 0.99),
                #         # 'conf_threshold_0.99_low_acc': (preds[u_idx][orig_conf < 0.99] == u_labels_[orig_conf < 0.99]).sum() / torch.sum(orig_conf < 0.99),
                #         # 'conf_threshold_0.8_ratio': torch.sum(orig_conf > 0.8) / len(u_labels_),
                #         # 'conf_threshold_0.8_high_acc': (preds[u_idx][orig_conf > 0.8] == u_labels_[orig_conf > 0.8]).sum() / torch.sum(orig_conf > 0.8),
                #         # 'conf_threshold_0.8_low_acc': (preds[u_idx][orig_conf < 0.8] == u_labels_[orig_conf < 0.8]).sum() / torch.sum(orig_conf < 0.8),
                #         # 'conf_threshold_0.9_ratio': torch.sum(orig_conf > 0.9) / len(u_labels_),
                #         # 'conf_threshold_0.9_high_acc': (preds[u_idx][orig_conf > 0.9] == u_labels_[orig_conf > 0.9]).sum() / torch.sum(orig_conf > 0.9),
                #         # 'conf_threshold_0.9_low_acc': (preds[u_idx][orig_conf < 0.9] == u_labels_[orig_conf < 0.9]).sum() / torch.sum(orig_conf < 0.9),
                #         'correct_avg_conf': orig_conf[preds[u_idx] == u_labels_].mean(),
                #         'incorrect_avg_conf': orig_conf[preds[u_idx] != u_labels_].mean(),
                #         'correct_avg_mcd_conf': mcd_conf[preds[u_idx] == u_labels_].mean(),
                #         'incorrect_avg_mcd_conf': mcd_conf[preds[u_idx] != u_labels_].mean(),
                #         'correct_conf': orig_conf[preds[u_idx] == u_labels_].detach().cpu().numpy(),
                #         'incorrect_conf': orig_conf[preds[u_idx] != u_labels_].detach().cpu().numpy(),
                #         'correct_mcd_conf': mcd_conf[preds[u_idx] == u_labels_].detach().cpu().numpy(),
                #         'incorrect_mcd_conf': mcd_conf[preds[u_idx] != u_labels_].detach().cpu().numpy(),
                #     })

        # Stochastic restore
        if conf.args.turn_off_reset is False and conf.args.dataset in ["tiny-imagenet", "imagenet"]:
            for nm, m in self.net.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        rand = torch.rand(p.shape)
                        mask = (rand < conf.args.restoration_factor).float().cuda()
                        with torch.no_grad():
                            p.data = self.src_net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

    def get_loss(self, mcd_softmax, original_softmax, labels, binary_feedbacks, len_c, len_w, len_u):
        correct_loss = torch.tensor([0.0]).to(device)
        wrong_loss = torch.tensor([0.0]).to(device)
        unlabeled_loss = torch.tensor([0.0]).to(device)

        cross_entropy = nn.CrossEntropyLoss(reduction='sum')

        if torch.sum(binary_feedbacks == 1) > 0:  # correct
            correct_index = binary_feedbacks == 1
            correct_dropout_outputs = mcd_softmax[correct_index]
            correct_labels = labels[correct_index]
            correct_loss = cross_entropy(correct_dropout_outputs, correct_labels) / len_c

        if torch.sum(binary_feedbacks == -1) > 0:  # wrong
            wrong_index = binary_feedbacks == -1
            wrong_dropout_outputs = mcd_softmax[wrong_index]
            wrong_labels = labels[wrong_index]
            wrong_loss = -cross_entropy(wrong_dropout_outputs, wrong_labels) / len_w

        if torch.sum(binary_feedbacks == 0) > 0:  # unlabeled
            unlabeled_index = binary_feedbacks == 0
            unlabeled_dropout_outputs = mcd_softmax[unlabeled_index]

            original_pred = original_softmax[unlabeled_index].argmax(dim=1).detach()

            same_pred_idx = original_pred == unlabeled_dropout_outputs.argmax(dim=1)
            if conf.args.ablation_conf_th > 0:
                same_pred_idx = original_softmax[torch.arange(len(original_pred)), original_pred] > conf.args.ablation_conf_th
            if conf.args.ablation_ent_th > 0:
                same_pred_idx = entropy(original_softmax[unlabeled_index]) < conf.args.ablation_ent_th

            unlabeled_loss = cross_entropy(unlabeled_dropout_outputs[same_pred_idx], original_pred[same_pred_idx]) / len_u

        return correct_loss, wrong_loss, unlabeled_loss

    def pre_active_sample_selection(self):
        self.enable_running_stats()
        self.disable_running_stats()

    def disable_running_stats(self):
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:  # use learn stats
                    module.track_running_stats = True
                    module.momentum = 0

    def enable_running_stats(self):
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:  # use learn stats
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
        self.net.train()
        feats, _, _ = self.fifo.get_memory()
        feats = torch.stack(feats).to(device)
        with torch.no_grad():
            _ = self.net(feats) # update bn stats
        pass


    def deepcopy_resnet50_domainnet(self, net):
        from models import ResNet
        copied_net = ResNet.ResNet50_DOMAINNET().to(device)
        copied_net = torch.nn.Sequential(copy.deepcopy(net[0]), copied_net)
        copied_net.train()

        for param in copied_net.parameters():  # turn on requires_grad for all
            param.requires_grad = True

        for name, module in copied_net.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'fc' in name:
                for param in module.parameters():
                    param.requires_grad = True
                continue

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats:  # use learn stats
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        copied_net.load_state_dict(copy.deepcopy(net.state_dict()))
        return copied_net