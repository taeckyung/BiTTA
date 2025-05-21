import os
from contextlib import nullcontext

import PIL
import math
import pandas as pd
import torch.nn
from torch import optim
from torch.nn import KLDivLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

import conf
from utils import cotta_utils, mecta
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
import copy
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class BiTTA(DNN):
    """
    Code is partially inspired from:
    - https://github.com/taeckyung/SoTTA/blob/main/learner/sotta.py
    - https://github.com/tmllab/2023_NeurIPS_FlatMatch/blob/main/trainer.py
    """

    def __init__(self, *args, **kwargs):
        self.atta_src_net = None
        super(BiTTA, self).__init__(*args, **kwargs)
        self.transform = get_tta_transforms()
        assert (conf.args.memory_type in ["ActivePriorityFIFO", "ActivePriorityPBRS"])

    def reset(self):
        super(BiTTA, self).reset()

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

        return optimizer

    def test_time_adaptation(self):
        # Unlabeled data
        u_feats, u_labels, _, _ = self.mem.get_u_memory()
        if len(u_feats) != 0:
            u_feats = torch.stack(u_feats).to(device)
            u_labels = torch.tensor(u_labels).to(device)
            u_dataset = torch.utils.data.TensorDataset(u_feats, u_labels)
            u_dataloader = DataLoader(u_dataset, batch_size=conf.args.update_every_x, #len(u_feats),
                                      shuffle=True, drop_last=False, pin_memory=False)
        else:
            u_dataloader = [(None, None)]

        # Correct data
        correct_feats, correct_labels, _, _ = self.mem.get_correct_memory()
        # print(len(correct_labels))
        self.json_active['num_correct_mem'] += [len(correct_labels)]
        if len(correct_feats) != 0:
            correct_feats = torch.stack(correct_feats).to(device)
            correct_labels = torch.tensor(correct_labels).to(device)

            correct_dataset = torch.utils.data.TensorDataset(correct_feats, correct_labels)
            correct_dataloader = DataLoader(correct_dataset, batch_size=conf.args.update_every_x, #len(correct_feats),
                                            shuffle=True, drop_last=False, pin_memory=False)
        else:
            correct_dataloader = [(None, None)]

        # Wrong data
        wrong_feats, wrong_labels, wrong_gt_labels, _ = self.mem.get_wrong_memory()
        self.json_active['num_wrong_mem'] += [len(wrong_labels)]
        if len(wrong_feats) != 0:
            wrong_feats = torch.stack(wrong_feats).to(device)
            wrong_labels = torch.tensor(wrong_labels).to(device)
            wrong_gt_labels = torch.tensor(wrong_gt_labels, dtype=torch.long).to(device)

            wrong_dataset = torch.utils.data.TensorDataset(wrong_feats, wrong_labels, wrong_gt_labels)
            wrong_dataloader = DataLoader(wrong_dataset, batch_size=conf.args.update_every_x, # len(wrong_feats),
                                          shuffle=True, drop_last=False, pin_memory=False)
        else:
            wrong_dataloader = [(None, None, None)]

        self.net.train()
        self.disable_running_stats()

        len_u_feats = len(u_feats) if conf.args.memory_size > 1 else 0
        if len(correct_feats) + len(wrong_feats) + len_u_feats == 0:
            return
            
        epoch = conf.args.epoch
        for e in range(epoch):
            for (u_feats_, u_labels_), (correct_feats_, correct_labels_), (wrong_feats_, wrong_labels_, wrong_gt_labels_) \
                    in zip(u_dataloader, correct_dataloader, wrong_dataloader):

                data = []
                if correct_feats_ is not None:
                    data.append(correct_feats_)
                if wrong_feats_ is not None:
                    data.append(wrong_feats_)
                if u_feats_ is not None:
                    data.append(u_feats_)
                assert(len(data) > 0)
                data = torch.cat(data, dim=0)

                # For optimization
                _, mcd_mean_softmax, _ = self.dropout_inference(data, conf.args.n_dropouts, dropout=conf.args.dropout_rate)

                with torch.no_grad():
                    outputs = self.net(data)
                    outputs_softmax = outputs.softmax(dim=1)

                # Get BFA and ABA losses for correct, incorrect, unlabeled samples
                correct_loss, wrong_loss, unlabeled_loss = self.get_loss(mcd_mean_softmax, outputs_softmax, correct_labels_, wrong_labels_, u_labels_)

                # Algorithm 1: Line 27~28 (Final update)
                loss = conf.args.w_final_loss_unlabeled * unlabeled_loss\
                       + (conf.args.w_final_loss_correct * correct_loss + conf.args.w_final_loss_wrong * wrong_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # Stochastic restoration for Tiny-ImageNet-C
        if conf.args.restoration_factor > 0:
            for nm, m in self.net.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        rand = torch.rand(p.shape)
                        mask = (rand < conf.args.restoration_factor).float().cuda()
                        with torch.no_grad():
                            p.data = self.src_net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

    def get_loss(self, mc_dropout_softmax, original_softmax, correct_labels_, wrong_labels_, u_labels_):
        correct_loss = torch.tensor([0.0]).to(device)
        wrong_loss = torch.tensor([0.0]).to(device)
        unlabeled_loss = torch.tensor([0.0]).to(device)

        # Algorithm 1: Line 20 (BFA loss)
        if correct_labels_ is not None:
            correct_dropout_outputs = mc_dropout_softmax[:len(correct_labels_)]
            if conf.args.use_original_conf:
                correct_dropout_outputs = original_softmax[:len(correct_labels_)]
            correct_loss = self.class_criterion(correct_dropout_outputs, correct_labels_)
        if wrong_labels_ is not None:
            start_idx = len(correct_labels_) if correct_labels_ is not None else 0
            end_idx = -len(u_labels_) if u_labels_ is not None else len(mc_dropout_softmax)

            own_wrong_dropout_outputs = mc_dropout_softmax[start_idx:end_idx] # softmax output of wrong sample
            wrong_loss = -self.class_criterion(own_wrong_dropout_outputs, wrong_labels_)

        # Algorithm 1: Line 22~25 (ABA loss)
        if u_labels_ is not None:
            start_idx = len(correct_labels_) if correct_labels_ is not None else 0
            start_idx += len(wrong_labels_) if wrong_labels_ is not None else 0

            own_u_dropout_outputs = mc_dropout_softmax[start_idx:]
            total_u_dropout_outputs = mc_dropout_softmax[start_idx:]

            original_pred = original_softmax[start_idx:].argmax(dim=1).detach()
            same_pred_idx = original_pred == total_u_dropout_outputs.argmax(dim=1)
            unlabeled_loss = self.class_criterion(own_u_dropout_outputs[same_pred_idx], original_pred[same_pred_idx])

        return correct_loss, wrong_loss, unlabeled_loss

    def pre_active_sample_selection(self):
        # Algorithm 1: Line 15
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


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    if conf.args.dataset in ['cifar10', 'cifar100', 'cifar10outdist', 'cifar100outdist']:
        img_shape = (32, 32, 3)
    else:
        img_shape = (224, 224, 3)

    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        cotta_utils.Clip(0.0, 1.0),
        cotta_utils.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        cotta_utils.GaussianNoise(0, gaussian_std),
        cotta_utils.Clip(clip_min, clip_max)
    ])
    return tta_transforms
