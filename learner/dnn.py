
from abc import abstractmethod

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
import webdataset as wds

import conf
from data_loader.data_loader import load_cache, save_cache
from models.ResNet import ResNetDropout18, ResNetDropout50
from models.ViT import vit_b_16
from utils.active_memory import ActivePriorityFIFO, ActivePriorityPBRS
from utils import memory, active_memory
from utils.calibration import expected_calibration_error
from utils.logging import *
from utils.loss_functions import *
from utils.memory import FIFO, ConfFIFO, HUS, Uniform, PBRS, CSTU
from utils.normalize_layer import *
import utils.reset_utils as reset_utils
import random
import time
import os
import data_loader.data_loader as data_loader_module

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

TRAINED = 0
SKIPPED = 1
FINISHED = 2


class DNN():
    def __init__(self, model_, corruption_list_):

        self.temp_value = 0
        self.count_bs1 = 63
        self.random_indexes_bs1 = np.arange(64)
        np.random.shuffle(self.random_indexes_bs1)
        self.random_indexes_bs1 = self.random_indexes_bs1[:3]
        
        self.count_skip = conf.args.enable_skip
        
        self.corruption_list = corruption_list_
        self.tgt_train_dist = conf.args.tgt_train_dist

        if conf.args.dataset == "imagenetR":
            filter_=conf.args.opt['indices_in_1k']
        else:
            filter_ = None


        ################## Init & prepare model###################
        # Load model
        if "pretrained" in conf.args.model:
            pretrained = model_(pretrained=True)
            if conf.args.model == "resnet18_pretrained":
                model = ResNetDropout18(filter=filter_)
                model.load_state_dict(pretrained.state_dict())
            elif conf.args.model == "resnet50_pretrained":
                model = ResNetDropout50(filter=filter_)
                model.load_state_dict(pretrained.state_dict())
            elif conf.args.model == "vitbase16_pretrained":
                model = vit_b_16()
                model.load_state_dict(pretrained.state_dict())
            else:
                raise NotImplementedError
            del pretrained
        else:
            if conf.args.model == "vitbase16":

                if conf.args.dataset == "cifar10":
                    model = model_(image_size=conf.args.opt['img_size'], num_classes=conf.args.opt['num_class'], patch_size=conf.args.vit_patch_size)
                else:
                    raise model_()
            
            elif conf.args.model == "resnet50_domainnet":
                model = model_()
            
            else:
                model = model_(filter=filter_)

        if conf.args.model in ['resnet50_domainnet']:
            self.net = model
        elif 'resnet' in conf.args.model:
            if conf.args.dataset in ["imagenet", "imagenetoutdist", "imagenetR"]:
                self.net = model
            else:
                num_feats = model.fc.in_features
                num_class = conf.args.opt['num_class']
                model.fc = nn.Linear(num_feats, num_class)  # match class number
                self.net = model

        elif conf.args.model in ["vitbase16", "vitbase16_pretrained"]:
            self.net = model

        if conf.args.load_checkpoint_path:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        norm_layer = get_normalize_layer(conf.args.dataset)

        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.net.to(device)

        ##########################################################

        # Important: some TTA methods would overwrite this optimizer
        self.optimizer = self.init_learner()

        self.class_criterion = nn.CrossEntropyLoss()

        ##################### enhance TTA ########################

        if conf.args.enhance_tta:
            self.enhance_tta()

        ##########################################################

        # online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'HUS':
            self.mem = memory.HUS(capacity=conf.args.memory_size, threshold=conf.args.high_threshold)
        elif conf.args.memory_type == 'CSTU':
            self.mem = memory.CSTU(capacity=conf.args.memory_size, num_class=conf.args.opt['num_class'],
                                         lambda_t=1, lambda_u=1)  # replace memory with original RoTTA
        elif conf.args.memory_type == 'ConfFIFO':
            self.mem = memory.ConfFIFO(capacity=conf.args.memory_size, threshold=conf.args.high_threshold)
        elif conf.args.memory_type == "ActivePriorityFIFO":
            self.mem = active_memory.ActivePriorityFIFO(conf.args.update_every_x, pop="", delay=conf.args.feedback_delay)
        elif conf.args.memory_type == "ActivePriorityPBRS":
            self.mem = active_memory.ActivePriorityPBRS(conf.args.update_every_x, pop="")
        else:
            raise NotImplementedError

        if conf.args.enable_bitta:
            self.active_mem = active_memory.ActivePriorityFIFO(conf.args.ass_num, pop="")
        else:
            self.active_mem = None

        self.fifo = memory.FIFO(conf.args.update_every_x)
        self.mem_state = self.mem.save_state_dict()
        self.net_state, self.optimizer_state = reset_utils.copy_model_and_optimizer(self.net, self.optimizer)

        self.num_batch_adapt = 0
        self.budget = 0

        # For BATTA
        self.rank = []
        self.rank_wrong = []
        self.temperature = 1.0
        self.num_correct = 0
        self.num_wrong = 0
        self.conf_sum = 0.0
        self.conf_correct_sum = 0.0

    @abstractmethod
    def init_learner(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            conf.args.opt['learning_rate'],
            momentum=conf.args.opt['momentum'],
            weight_decay=conf.args.opt['weight_decay'],
            nesterov=True)
        return optimizer

    @abstractmethod
    def test_time_adaptation(self):
        assert isinstance(self.mem, FIFO)
        feats, labels, _ = self.mem.get_memory()
        feats = torch.stack(feats).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(feats, labels)
        data_loader = DataLoader(dataset, batch_size=conf.args.tta_batch_size,
                                 shuffle=True, drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats, _) in enumerate(data_loader):
                if len(feats) == 1:
                    self.net.eval()  # avoid BN error
                else:
                    self.net.train()

                if conf.args.method in ['Src']:
                    pass
                else:
                    raise NotImplementedError

    @abstractmethod
    def run_before_training(self):
        pass

    def reset(self):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net, self.optimizer, self.net_state, self.optimizer_state)
        self.mem.reset()

    def init_json(self, log_path):
        self.write_path = log_path
        self.json_eval = {
            k: [] for k in ['gt', 'accuracy', 'current_accuracy',
                            'pred', 'confidence', 'entropy','energy','grad',
                            'dropout_confidence', 'dropout_01_confidence',
                            'original_ebce', 'dropout_ebce', 'cumul_original_ebce', 'cumul_dropout_ebce', 'cumul_dropout_01_ebce']
        }

        self.json_active = {
            k: [] for k in ["budgets", "correct_loss", "wrong_loss", "unlabel_loss",
                            "num_correct_mem", "num_wrong_mem"]
        }
    
    def set_target_data(self, source_data_loader, source_val_data_loader, target_data_loader, corruption):
        self.source_dataloader = source_data_loader
        self.source_val_dataloader = source_val_data_loader
        self.target_dataloader = target_data_loader

        dataset = conf.args.dataset
        cond = corruption

        filename = f"{dataset}_{conf.args.seed}_dist{conf.args.tgt_train_dist}"

        if conf.args.tgt_train_dist == 4:
            filename += f"_gamma{conf.args.dirichlet_beta}"

        file_path = conf.args.opt['file_path'] + "_target_train_set"

        self.target_train_set = load_cache(filename, cond, file_path, transform=None)

        if not self.target_train_set:
            self.target_data_processing()
            save_cache(self.target_train_set, filename, cond, file_path, transform=None)
        
        if conf.args.save_wds_dataset:
            try:
                if not os.path.exists("cache_wds_cifar10_random_setting"):
                    os.makedirs("cache_wds_cifar10_random_setting")
            except:
                pass
            sink = wds.TarWriter(os.path.join(f"cache_wds_cifar10_random_setting/{filename}_{cond}" + ".tar"))
            
            for index in range(len(self.target_train_set[0])):
                input, output, dls = torchvision.transforms.functional.to_pil_image(self.target_train_set[0][index]), self.target_train_set[1][index], self.target_train_set[2][index]
                sink.write(
                    {
                        "__key__": "sample_" + str(index),
                        "info": "",
                        "input.jpg": input,
                        "output.cls": output.item(),
                        "dls.cls": dls.item(),
                    }
                )

            sink.close()

    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):
            # must be loaded from dataloader, due to transform in the __getitem__()

            features.append(feat.squeeze(0))
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        # num_class = conf.args.opt['num_class']

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        tgt_train_dist_ = self.tgt_train_dist
        # real distribution
        if tgt_train_dist_ == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        elif tgt_train_dist_ == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # dirichlet distribution
        elif self.tgt_train_dist == 4:
            dirichlet_numchunks = conf.args.opt['num_class']
            num_class = conf.args.opt['num_class']

            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
            min_size = -1
            N = len(features)
            min_size_thresh = 10 #if conf.args.dataset in ['tinyimagenet'] else 10
            while min_size < min_size_thresh:  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
                for k in range(num_class):
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    # balance
                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []

            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(num_class))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    result_feats.extend([features[i] for i in idx])
                    result_cl_labels.extend([cl_labels[i] for i in idx])
                    result_do_labels.extend([do_labels[i] for i in idx])
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            # trim data if num_sample is smaller than the original data size
            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        else:
            raise NotImplementedError

        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            try:
                self.target_train_set = (torch.stack(result_feats),
                                         result_cl_labels,
                                         torch.stack(result_do_labels))
            except:  # for dataset which each image has different shape
                self.target_train_set = (result_feats,
                                         result_cl_labels,
                                         torch.stack(result_do_labels))

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer) or isinstance(self.net[0], IdentityLayer):
                cp = self.net[1]
            else:
                raise NotImplementedError

        else:
            cp = self.net

        if isinstance(self.net, nn.DataParallel):
            cp = self.net.module

        torch.save(cp.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path.split(".")[-1] == ("pickle"):
            import pickle
            with open(os.path.join(checkpoint_path), 'rb') as f:
                loaded_net = pickle.load(f)

            if 'resnet18' in conf.args.model:
                if conf.args.dataset == "colored-mnist":
                    self.net = ResNetDropout18()

                    num_feats = self.net.fc.in_features
                    num_class = conf.args.opt['num_class']
                    self.net.fc = nn.Linear(num_feats, num_class)  # match class number
                    self.net = self.net

                    self.net.load_state_dict(loaded_net.state_dict())
                    self.net.to(device)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        else:
            self.checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
            if conf.args.dataset == "domainnet-126":
                temp_dict = {}
                for k, v in self.checkpoint['state_dict'].items():
                    keywords = k.split(".")[1:]
                    for i in range(1, len(keywords)):
                        if keywords[i-1] == "encoder":
                            if keywords[i] == "1":
                                keywords[i-1] = "bn_encoder"
                            keywords = keywords[:i] + keywords[i+1:]
                            break

                    target_k = ".".join(keywords)

                    temp_dict[target_k] = v
                self.checkpoint = temp_dict
            self.net.load_state_dict(self.checkpoint, strict=True)
            self.net.to(device)

    def log_loss_results(self, condition, epoch, loss_avg, train_acc_avg= None, val_loss_avg=None, val_acc_avg=None):

        if condition == 'train_online':
            # print loss
            print('{:s}: [current_sample: {:d}]'.format(
                condition, epoch
            ))
        else:
            # print loss
            text = '{:s}: [epoch: {:d}]\tLoss: {:.6f} \t'.format(condition, epoch, loss_avg)
            if train_acc_avg != None:
                text += 'TrainAcc: {:.6f} \t'.format(train_acc_avg)
            if val_loss_avg != None:
                text += 'val_loss_avg: {:.6f} \t'.format(val_loss_avg)
            if val_acc_avg != None:
                text += 'val_acc_avg: {:.6f} \t'.format(val_acc_avg)
            print(text)

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])
        # assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """
        # setup models

        self.net.train()

        class_loss_sum = 0.0
        class_loss_sum_val = 0.0

        total_iter = 0
        total_iter_val = 0

        total_num_correct = 0
        total_num_correct_val = 0

        total_num_samples = len(self.source_dataloader['train']) * conf.args.opt['batch_size']
        total_num_samples_val= 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                if len(cls.shape) > 1:
                    cls = cls.flatten()
                feats, cls = feats.to(device), cls.to(device)

                # compute the feature
                preds = self.net(feats)
                with torch.no_grad():
                    pred_cls = preds.max(axis = 1).indices
                    num_correct_preds = (cls == pred_cls).sum()
                    total_num_correct += num_correct_preds
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()

            if self.source_dataloader['valid'] != None:
                total_num_samples_val = len(self.source_dataloader['valid']) * conf.args.opt['batch_size']
                with torch.no_grad():
                    num_iter_val = len(self.source_dataloader['valid'])
                    total_iter_val += num_iter_val
                    for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['valid']), total=num_iter_val):
                        feats, cls, _ = labeled_data
                        if len(cls.shape) > 1:
                            cls = cls.flatten()
                        feats, cls = feats.to(device), cls.to(device)

                        # compute the feature
                        preds = self.net(feats)
                        pred_cls = preds.max(axis = 1).indices
                        num_correct_preds = (cls == pred_cls).sum()
                        total_num_correct_val += num_correct_preds
                        class_loss_val = self.class_criterion(preds, cls)
                        class_loss_sum_val += float(class_loss_val * feats.size(0))

        dict_wandb_log = {
                        'epoch' : epoch,
                        'train_loss': class_loss_sum/total_iter,
                        'val_loss': class_loss_sum_val/total_iter_val if total_iter_val != 0 else 0,
                        'train_acc' : total_num_correct/total_num_samples,
                        'val_acc' : total_num_correct_val/total_num_samples_val if total_num_samples_val != 0 else 0
                    }

        if conf.args.wandb:
            import wandb
            wandb.log(
                dict_wandb_log
            )

        # Logging
        self.log_loss_results('train', epoch=epoch, 
                              loss_avg=class_loss_sum / total_iter,
                              train_acc_avg=total_num_correct/total_num_samples,
                              val_loss_avg=class_loss_sum_val/total_iter_val if total_iter_val != 0 else 0,
                              val_acc_avg=total_num_correct_val/total_num_samples_val if total_num_samples_val != 0 else 0,
        )
        avg_loss = class_loss_sum / total_iter



        return avg_loss

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')

    def evaluation(self, epoch, condition):
        # Evaluate with a batch of samples, which is a typical way of evaluation. Used for pre-training or offline eval.

        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)

        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, current_samples):
        # Evaluate with online samples that come one by one while keeping the order.
        self.net.eval()

        with torch.no_grad():  # we don't log grad here
            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples
            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            self.evaluation_online_body(epoch, current_samples, feats, cls, dls)

    def model_inference(self, feats, net=None, temp=1.0, get_embedding=False, eval_mode=True):
        if net is None:
            net = self.net

        if eval_mode:
            net.eval()
        else:
            net.train()

        len_feats = len(feats)

        if len(feats) == 1:
            if not conf.args.use_learned_stats:
                feats = torch.concat([feats, feats])
                len_feats = 1
            else:
                net.eval()

        # Normalization layer: self.net[0] / ResNet: self.net[1]
        if get_embedding:
            y_logit, y_embedding = net[1](net[0](feats), get_embedding=True)
            y_embedding = y_embedding[:len_feats]

        else:
            y_logit = net(feats)
            y_embedding = None

        y_logit = y_logit[:len_feats]


        y_entropy: torch.Tensor = softmax_entropy(y_logit)
        y_pred_softmax: torch.Tensor = F.softmax(y_logit / temp, dim=1)
        y_conf: torch.Tensor = y_pred_softmax.max(1, keepdim=False)[0]
        y_energy: torch.Tensor = calc_energy(y_logit).cpu()
        y_pred: torch.Tensor = y_logit.max(1, keepdim=False)[1]

        return y_pred, y_conf, y_entropy, y_energy, y_embedding, y_pred_softmax, y_logit

    def evaluation_online_body(self, epoch, current_samples, feats, cls, dls):
        # get lists from json

        true_cls_list = self.json_eval['gt']
        pred_cls_list = self.json_eval['pred']
        accuracy_list = self.json_eval['accuracy']
        conf_list = self.json_eval['confidence']
        entropy_list = self.json_eval['entropy']
        current_accuracy_list = self.json_eval['current_accuracy']
        dropout_conf_list = self.json_eval['dropout_confidence']
        dropout_01_conf_list = self.json_eval['dropout_01_confidence']
        original_ebce_list = self.json_eval['original_ebce']
        dropout_ebce_list = self.json_eval['dropout_ebce']
        cumul_original_ebce_list = self.json_eval['cumul_original_ebce']
        cumul_dropout_ebce_list = self.json_eval['cumul_dropout_ebce']
        cumul_dropout_01_ebce_list = self.json_eval['cumul_dropout_01_ebce']

        cls = cls.to(torch.int32)
        feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

        # Inference
        y_pred, y_conf, y_entropy, y_energy, y_embeddings, y_logit, y_output = self.model_inference(feats)

        # append values to lists
        current_true_cls_list = [int(c) for c in cls.tolist()]
        true_cls_list += current_true_cls_list
        current_pred_cls_list = [int(c) for c in y_pred.tolist()]
        pred_cls_list += current_pred_cls_list
        conf_list += [float(c) for c in y_conf.tolist()]
        entropy_list += [float(c) for c in y_entropy.tolist()]

        if conf.args.dropout_rate != -1:
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(feats, n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                _, dropout_01_softmax_mean, _ = self.dropout_inference(feats, n_iter=1, dropout=conf.args.dropout_rate, net=self.net)
                dropout_conf_for_pred = dropout_softmax_mean[:, y_pred].diagonal()
                dropout_01_conf_for_pred = dropout_01_softmax_mean[:, y_pred].diagonal()
                dropout_conf_list += [float(c) for c in dropout_conf_for_pred]
                dropout_01_conf_list += [float(c) for c in dropout_01_conf_for_pred]

            original_ebce, _ = expected_calibration_error(y_conf, y_pred==cls, num_bins=10, order=2)
            dropout_ebce, _ = expected_calibration_error(dropout_conf_for_pred, y_pred==cls, num_bins=10, order=2)

            cumul_correct = torch.Tensor(true_cls_list)==torch.Tensor(pred_cls_list)
            cumul_original_ebce, _ = expected_calibration_error(torch.Tensor(conf_list), cumul_correct, num_bins=10, order=2)
            cumul_dropout_ebce, _ = expected_calibration_error(torch.Tensor(dropout_conf_list), cumul_correct, num_bins=10, order=2)
            cumul_dropout_01_ebce, _ = expected_calibration_error(torch.Tensor(dropout_01_conf_list), cumul_correct, num_bins=10, order=2)

            original_ebce_list.append(original_ebce)
            dropout_ebce_list.append(dropout_ebce)
            cumul_original_ebce_list.append(cumul_original_ebce)
            cumul_dropout_ebce_list.append(cumul_dropout_ebce)
            cumul_dropout_01_ebce_list.append(cumul_dropout_01_ebce)

            original_conf_gt_class = y_logit[:, cls.to(torch.long)].diagonal()
            dropout_conf_gt_class = dropout_softmax_mean[:, cls.to(torch.long)].diagonal()

        if len(true_cls_list) > 0:
            current_accuracy = sum(1 for gt, pred in zip(current_true_cls_list, current_pred_cls_list) if gt == pred) \
                               / float(len(current_true_cls_list)) * 100
            current_accuracy_list.append(current_accuracy)
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) \
                             / float(len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)

            
            dict_wandb_log = {
                        'num_batch_adapt': self.num_batch_adapt,
                        'accuracy': cumul_accuracy,
                        'current_accuracy': current_accuracy,
                        'entropy': y_entropy.mean().item(),
                        'confidence': y_conf.mean().item(),
                        'energy': y_energy.mean().item(),
                        'original_ebce': original_ebce,
                        'dropout_ebce': dropout_ebce,
                        'cumul_original_ebce': cumul_original_ebce,
                        'cumul_dropout_ebce': cumul_dropout_ebce,
                        'cumul_dropout_01_ebce': cumul_dropout_01_ebce,
                        # 'entropy_correct': y_entropy[y_pred == cls],
                        # 'entropy_wrong': y_entropy[y_pred != cls],
                        # 'original_conf_correct': original_conf_gt_class[y_pred == cls],
                        # 'dropout_conf_correct': dropout_conf_gt_class[y_pred == cls],
                        # 'original_conf_wrong': original_conf_gt_class[y_pred != cls],
                        # 'dropout_conf_wrong': dropout_conf_gt_class[y_pred != cls],
                        # 'conf_diff_correct': original_conf_gt_class[y_pred == cls] - dropout_conf_gt_class[y_pred == cls],
                        # 'conf_diff_wrong': original_conf_gt_class[y_pred != cls] - dropout_conf_gt_class[y_pred != cls]
                        # 'entropy_ema': self.entropy_ema,
                        # 'entropy_diff': self.entropy_diff,
                        # 'dropout_conf_correct': dropout_conf_for_pred[y_pred == cls],
                        # 'dropout_conf_wrong': dropout_conf_for_pred[y_pred != cls],
                        # 'dropout_ebce': dropout_ebce,
                        # 'original_ebce': original_ebce,
                        # 'cumul_dropout_ebce': cumul_dropout_ebce,
                        # 'cumul_original_ebce': cumul_original_ebce,
                        # 'original_conf_gt_class': original_conf_gt_class,
                        # 'dropout_conf_gt_class': dropout_conf_gt_class,
                        # 'original_mean_conf_gt_class': original_conf_gt_class.mean(),
                        # 'dropout_mean_conf_gt_class': dropout_conf_gt_class.mean(),
                    }

            if conf.args.wandb:
                import wandb
                wandb.log(
                    dict_wandb_log
                )

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]
            

            # epoch: 1~len(self.target_train_set[0])
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if conf.args.wds_path is not None:
                    if i % conf.args.update_every_x == 0:
                        print(
                            f'[Online Eval][NumSample:{i}][Epoch:{i}][Accuracy:{cumul_accuracy}]')
                else:
                    if i in progress_checkpoint:
                        print(
                            f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

        # update self.json file
        self.json_eval['gt'] = true_cls_list
        self.json_eval['pred'] = pred_cls_list
        self.json_eval['accuracy'] = accuracy_list
        self.json_eval['confidence'] = conf_list
        self.json_eval['entropy'] = entropy_list
        self.json_eval['current_accuracy'] = current_accuracy_list
        self.json_eval['dropout_confidence'] = dropout_conf_list
        self.json_eval['original_ebce'] = original_ebce_list
        self.json_eval['dropout_ebce'] = dropout_ebce_list
        self.json_eval['cumul_original_ebce'] = cumul_original_ebce_list
        self.json_eval['cumul_dropout_ebce'] = cumul_dropout_ebce_list

    def dump_eval_online_result(self, is_train_offline=False):
        if is_train_offline:
            if conf.args.wds_path is not None:
                count_num_samples = 0
                while True:
                    try:
                        self.target_train_set = self.iter_target_train_set.next()
                        self.target_train_set[1] = self.target_train_set[1]
                    except:
                        break
                    
                    feats, cls, dls = self.target_train_set
                    current_sample = feats, cls, dls      
                    count_num_samples += len(feats)
                    self.evaluation_online(count_num_samples,
                                        [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])
            else:
                feats, cls, dls = self.target_train_set
                batchsize = conf.args.opt['batch_size']
                for num_sample in range(0, len(feats), batchsize):
                    current_sample = feats[num_sample:num_sample + batchsize], cls[num_sample:num_sample + batchsize], dls[
                                                                                                                    num_sample:num_sample + batchsize]
                    self.evaluation_online(num_sample + batchsize,
                                        [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json = self.json_eval | self.json_active
        json_subsample = {key: json[key] for key in json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss

    def add_instance_to_memory(self, current_sample, mem):
        with torch.no_grad():
            self.net.eval()

            if isinstance(mem, FIFO):
                mem.add_instance(current_sample)


            else:
                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                y_pred, y_conf, y_entropy, y_energy, y_embeddings, y_pred_softmax, _ = self.model_inference(
                    f.unsqueeze(0))

                if isinstance(mem, ConfFIFO) or isinstance(mem, HUS) or isinstance(mem, Uniform) or isinstance(mem, PBRS):
                    mem.add_instance([f, y_pred.item(), d, y_conf.item(), c.item()])

                elif isinstance(mem, CSTU):
                    mem.add_instance([f, y_pred.item(), y_entropy.item(), c.item()])

                elif isinstance(mem, ActivePriorityFIFO):
                    mem.add_u_instance([f, c.item(), d, y_entropy.item()])
                
                elif isinstance(mem, ActivePriorityPBRS):
                    mem.add_u_instance([f, c.item(), d, y_entropy.item(), y_pred.item()])
                else:
                    raise NotImplementedError

    def train_online(self, current_num_sample):
        """
        Train the model
        """
        if conf.args.wds_path is not None:
            if self.target_train_set is None:
                current_num_sample_in_batch = 0
            else:
                current_num_sample_in_batch = (current_num_sample - 1) % len(self.target_train_set[0])
            if current_num_sample_in_batch == 0:
                try:
                    self.target_train_set = self.iter_target_train_set.next()
                    # self.target_train_set[1] = self.target_train_set[1]
                except Exception as error:
                    print("An exception occurred:", error)
                    # f = open("ccc_error.txt", "a")
                    # f.write(str(error))
                    # f.close()
                    return FINISHED
            current_sample = self.target_train_set[0][current_num_sample_in_batch], self.target_train_set[1][current_num_sample_in_batch], torch.tensor([0.0])
        else:
            if current_num_sample > len(self.target_train_set[0]):
                return FINISHED

            # batch_data, cls = self.temp_dataloader[self.temp_dataloader_id]
            # current_sample = batch_data[current_num_sample - 1], cls[current_num_sample - 1], torch.tensor([0.0])

            
            #-- comment debug
            # # Add a sample
            batch_data, cls, dls = self.target_train_set
            current_sample = batch_data[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        #-#
        # print("temp")
        self.add_instance_to_memory(current_sample, self.fifo)  # for evaluation
        self.add_instance_to_memory(current_sample, self.mem)  # for test-time adaptation
        if conf.args.enable_bitta:
            self.add_instance_to_memory(current_sample, self.active_mem)

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[0]) and
                    conf.args.update_every_x >= current_num_sample):  # update with entire data
                # self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)
                return SKIPPED

        self.evaluation_online(current_num_sample, self.fifo.get_memory())

        if conf.args.no_adapt:  # for ablation
            return TRAINED
        
        self.pre_active_sample_selection()
        
        if isinstance(self.mem, ActivePriorityFIFO) or isinstance(self.mem, ActivePriorityPBRS):
            self.active_sample_selection(self.mem, current_num_sample)
        elif conf.args.enable_bitta:
            self.active_sample_selection(self.active_mem, current_num_sample)

        prev_wall_time = time.time()
        self.test_time_adaptation()
        curr_wall_time = time.time()

        if conf.args.wandb:
            import wandb
            wandb.log({
                'num_batch_adapt': self.num_batch_adapt,
                'wall_clock_time_per_batch': curr_wall_time - prev_wall_time
            })

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        self.num_batch_adapt += 1

        return TRAINED

    @abstractmethod
    def pre_active_sample_selection(self):
        pass

    def dropout_inference(self, x, n_iter, dropout, net=None, temperature=1.0):
        if net is None:
            net = self.net
        net = self.net.module if isinstance(self.net, nn.DataParallel) or isinstance(self.net, nn.parallel.DistributedDataParallel) else net

        if dropout < 0:
            if conf.args.dataset == "pacs":
                dropout = 0.4
            elif conf.args.dataset == "vlcs":
                dropout = 0.3
            elif conf.args.dataset == "tiny-imagenet":
                dropout = 0.3
            elif conf.args.dataset == "cifar10":
                dropout = 0.3
            else:
                raise NotImplementedError

        predictions = []
        for _ in range(n_iter):
            pred = net[1]((net[0](x)), dropout=dropout)  # batch_size, n_classes
            pred = F.softmax(pred, dim=1) / temperature
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # batch_size, n_iter, n_classes
        pred_class = torch.argmax(predictions, dim=2)
        mean_pred = torch.mean(predictions, dim=1)
        mean_pred_class = torch.argmax(mean_pred, dim=1)
        std_pred = torch.std(predictions, dim=1)
        return predictions, mean_pred, std_pred

    def active_sample_selection(self, mem, current_num_sample):
        assert isinstance(mem, ActivePriorityFIFO) or isinstance(mem, ActivePriorityPBRS)
    
        if conf.args.memory_size == 1:
            self.count_bs1 += 1
            self.count_bs1 %= 64
            
            if self.count_bs1 == 0:
                self.random_indexes_bs1 = np.arange(64)
                np.random.shuffle(self.random_indexes_bs1)
                self.random_indexes_bs1 = self.random_indexes_bs1[:3]

            if self.count_bs1 not in self.random_indexes_bs1:
                return
            
        selected_feats, selected_labels, selected_domains = [], [], []

        ass_num = conf.args.ass_num
        if conf.args.ass_per_n_step:
            if self.num_batch_adapt % conf.args.ass_per_n_step != 0:
                return

        self.net.train()  # network is not updating BN stats (is disabled); just using test statistics

        if isinstance(mem, ActivePriorityFIFO) or isinstance(mem, ActivePriorityPBRS):
            feats, gt_labels, domains, entropies = mem.get_u_memory()
        else:
            raise NotImplementedError
        # selected_index = torch.topk(torch.Tensor(entropies), ass_num).indices

        if conf.args.sample_selection == "conf_diff":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
                original_confidences = y_conf
            selected_index = torch.topk(original_confidences - dropout_confidences, ass_num, largest=True).indices
        elif conf.args.sample_selection == "agree_conf_diff":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
                original_confidences = y_conf
                agreement = y_pred == dropout_softmax_mean.argmax(dim=1)
                original_confidences[~agreement] = 0.0
            selected_index = torch.topk(original_confidences - dropout_confidences, ass_num, largest=True).indices
        elif conf.args.sample_selection == "random":
            selected_index = torch.randperm(len(gt_labels))[:ass_num]
        elif conf.args.sample_selection == "entropy":
            selected_index = torch.topk(torch.Tensor(entropies), ass_num).indices
        elif conf.args.sample_selection == "conf":
            with torch.no_grad():
                _, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
            selected_index = torch.topk(y_conf, ass_num, largest=False).indices
        elif conf.args.sample_selection == "energy":
            with torch.no_grad():
                _, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
            selected_index = torch.topk(y_energy, ass_num, largest=False).indices
        elif conf.args.sample_selection == "mc_entropy":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                dropout_entropies = Entropy(dropout_softmax_mean)
            selected_index = torch.topk(dropout_entropies, ass_num, largest=True).indices
        elif conf.args.sample_selection == "mc_conf":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                # dropout_confidences = dropout_softmax_mean.max(dim=1)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
            selected_index = torch.topk(dropout_confidences, ass_num, largest=False).indices
        elif conf.args.sample_selection == "mc_conf_agree":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                # dropout_confidences = dropout_softmax_mean.max(dim=1)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
            agreement = y_pred != dropout_softmax_mean.argmax(dim=1)

            # Step 1: Mask the confidence tensor where binary label is True
            masked_confidence = dropout_confidences[agreement]

            # Step 2: Get the top 3 indices in the masked confidence tensor
            top3_values, top3_indices = torch.topk(masked_confidence, k=ass_num, largest=False)

            # Step 3: Convert the indices of the masked tensor back to the original tensor's indices
            original_indices = torch.nonzero(agreement).squeeze(1)  # Get original indices where binary is True
            selected_index = original_indices[top3_indices]

            if len(selected_index) < ass_num:
                selected_index = torch.topk(dropout_confidences, ass_num, largest=False).indices
        elif conf.args.sample_selection == "mc_disagree":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
            disagree = y_pred != dropout_softmax_mean.argmax(dim=1)
            true_indices = torch.nonzero(disagree, as_tuple=True)[0]
            selected_index = true_indices[torch.randperm(len(true_indices))[:ass_num]]
            if len(selected_index) < ass_num:
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
                selected_index = torch.topk(dropout_confidences, ass_num, largest=False).indices
        elif conf.args.sample_selection == "mc_energy":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                energy = calc_energy(dropout_softmax_mean).cpu()
            selected_index = torch.topk(energy, ass_num, largest=False).indices
        elif conf.args.sample_selection == "mc_variance":
            with torch.no_grad():
                _, dropout_softmax_mean, dropout_softmax_var = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                max_class = dropout_softmax_mean.max(1, keepdim=False)[1]
                variance = dropout_softmax_var[:, max_class].diagonal()
            selected_index = torch.topk(variance, ass_num, largest=True).indices
        elif conf.args.sample_selection == "uncertainty":
            mcd_softmaxs, mcd_mean_softmax, _ = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate)
            mcd_mean_expanded = mcd_mean_softmax.unsqueeze(1).expand_as(mcd_softmaxs)
            epistemic_uncertainty = ((mcd_softmaxs - mcd_mean_expanded) ** 2).mean(dim=1).sum(dim=1)  # variance over mc-dropouts
            aleatoric_uncertainty = entropy(mcd_mean_softmax)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            selected_index = torch.topk(total_uncertainty, ass_num, largest=True).indices

        elif conf.args.sample_selection == "max_conf":
            with torch.no_grad():
                _, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
            selected_index = torch.topk(y_conf, 10, largest=True, sorted=False).indices[:ass_num]
        elif conf.args.sample_selection == "min_conf":
            with torch.no_grad():
                _, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
            selected_index = torch.topk(y_conf, 10, largest=False, sorted=False).indices[:ass_num]
        
        elif conf.args.sample_selection == "mc_conf_stable":
            with torch.no_grad():
                _, dropout_softmax_mean, _ = self.dropout_inference(torch.stack(feats), n_iter=10, dropout=conf.args.dropout_rate, net=self.net)
                dropout_confidences = dropout_softmax_mean.max(dim=1).values
            selected_index = torch.topk(dropout_confidences, 10, largest=False).indices[:ass_num]
            
        else:
            raise NotImplementedError
     

        self.budget += ass_num
        print("budget = ", self.budget)
        self.json_active['budgets'] += [self.budget]

        if conf.args.wandb:
            import wandb
            wandb.log(
                {
                    'num_batch_adapt': self.num_batch_adapt,
                    'budget': self.budget,
                }
            )
                
        # add active samples to memory
        if ass_num <= 0:
            return

        selected_index = selected_index.sort(descending=True).values
        for idx in selected_index:
            selected_feats += [feats[idx]]
            selected_labels += [gt_labels[idx]]
            selected_domains += [domains[idx]]

            mem.remove_u_instance_by_index(idx)  # remove index from candidate pool

        if not conf.args.active_binary:  # full active TTA
            for correct_data_i in range(len(selected_feats)):
                data = [selected_feats[correct_data_i],
                        selected_labels[correct_data_i],
                        selected_domains[correct_data_i],
                        0.0]
                mem.add_correct_instance(data)
            return

        # binary TTA
        self.net.eval()
        selected_feats_ = torch.stack(selected_feats).to(device)
        selected_labels_ = torch.Tensor(selected_labels).to(device)
        negative_cls = []


        with torch.no_grad():
            y_logit = self.net(selected_feats_)
            y_entropy = softmax_entropy(y_logit)
            y_pred = y_logit.max(1, keepdim=False)[1]
            for y_logit_ in y_logit:
                negative_cls += [y_logit_.softmax(0).sort(descending = False).indices[:conf.args.ass_aug_negative].tolist()]

        if conf.args.save_img:
            for selected_feats_ii in range(len(selected_feats_)):
                dir_path = f"save_img/{conf.args.current_corruption[:-2]}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torchvision.utils.save_image(selected_feats_[selected_feats_ii], f"{dir_path}/{self.temp_value}_gt{int(selected_labels_[selected_feats_ii])}_pred{y_pred[selected_feats_ii]}.jpg")
                self.temp_value += 1

        mask_correct: torch.Tensor = y_pred == selected_labels_

        for match_i in range(len(mask_correct)):
            label = selected_labels_[match_i].item()
            if conf.args.label_error_type == 'symmetric' and conf.args.label_error_rate > 0:
                if np.random.random() < conf.args.label_error_rate:  # select noisy label
                    indices = list(range(conf.args.opt['num_class']))
                    indices.remove(label)
                    label = random.choice(indices)
            if conf.args.label_error_type == 'asymmetric' and conf.args.label_error_rate > 0:
                if np.random.random() < conf.args.label_error_rate:  # select noisy label
                    if conf.args.dataset == "cifar10":
                        # TRUCK -> AUTOMOBILE, BIRD -> PLANE, DEER -> HORSE, and CAT <-> DOG
                        if label == 9: label = 1
                        elif label == 2: label = 0
                        elif label == 3: label = 5
                        elif label == 5: label = 3
                        elif label == 5: label = 7
                    elif conf.args.dataset == "cifar100":
                        # shift one next in the superclass
                        base = (label // 5) * 5
                        new_index = (label % 5 + 1) % 5
                        label = base + new_index
                    else:
                        raise NotImplementedError
            if conf.args.feedback_error_rate > 0 and np.random.random() < conf.args.feedback_error_rate:
                mask_correct[match_i] = ~mask_correct[match_i]

            if conf.args.ass_aug_negative > 0:
                if mask_correct[match_i]:
                    data = [selected_feats[match_i],
                        y_pred[match_i].item(),
                        label,
                        y_entropy[match_i].item()]
                else:
                    concat_class = negative_cls[match_i]
                    if y_pred[match_i].item() in concat_class:
                        concat_class.remove(y_pred[match_i].item())
                    concat_class = [y_pred[match_i].item()] + concat_class
                    data = [selected_feats[match_i],
                        concat_class,
                        label,
                        y_entropy[match_i].item()]

            else:
                data = [selected_feats[match_i],
                        y_pred[match_i].item(),
                        label,
                        y_entropy[match_i].item()]

            if mask_correct[match_i]:  # correct
                mem.add_correct_instance(data)
                self.num_correct += 1
            else:  # wrong
                mem.add_wrong_instance(data)
                indices_sorted = torch.argsort(y_logit.softmax(dim=1)[match_i], descending=True)
                rank = (indices_sorted == selected_labels_[match_i].item()).nonzero().item()
                self.rank_wrong.append(rank)
                if conf.args.wandb:
                    import wandb
                    wandb.log({
                        'num_batch_adapt': self.num_batch_adapt,
                        'wrong_rank': self.rank_wrong
                    })
                self.num_wrong += 1

            indices_sorted = torch.argsort(y_logit.softmax(dim=1)[match_i], descending=True)
            rank = (indices_sorted == selected_labels_[match_i].item()).nonzero().item()
            self.rank.append(rank)
            if conf.args.wandb:
                import wandb
                wandb.log({
                    'num_batch_adapt': self.num_batch_adapt,
                    'rank': self.rank,
                    'num_correct': self.num_correct,
                    'num_wrong': self.num_wrong
                })

    def enhance_tta(self):
        from utils.loss_functions import complement_CrossEntropyLoss

        if conf.args.model in ["resnet18_pretrained", "resnet18"]:
            if conf.args.dataset == "pacs":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 150
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            elif conf.args.dataset == "vlcs":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 25
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            elif conf.args.dataset == "tiny-imagenet":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 25
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            elif conf.args.dataset == "cifar10":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 150
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            elif conf.args.dataset == "cifar100":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 150
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            elif conf.args.dataset == 'imagenet':
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 20
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            else:
                pass # use default value
        elif conf.args.model in ['resnet50_pretrained', 'resnet50']:
            if conf.args.dataset == "pacs":
                conf.args.enhance_tta_lr = 0.00001
                conf.args.enhance_tta_epoch = 5
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 0.1
            else:
                pass
        elif conf.args.model in ['vitbase16']:
            if conf.args.dataset == "cifar10":
                conf.args.enhance_tta_lr = 0.001
                conf.args.enhance_tta_epoch = 20
                conf.args.w_final_loss_correct = 1.0
                conf.args.w_final_loss_wrong = 1.0
            else:
                pass
        else:
            pass


        optimizer = torch.optim.SGD(
            self.net.parameters(),
            conf.args.enhance_tta_lr,
            momentum=conf.args.opt['momentum'],
            weight_decay=conf.args.opt['weight_decay'],
            nesterov=True)

            
        # original_result_path, original_checkpoint_path, original_log_path = self.paths

        correct_active_feats, correct_active_cls, correct_active_dls = [],[],[]
        wrong_active_feats, wrong_active_cls, wrong_active_dls = [],[],[]

        self.disable_running_stats()

        dataset = conf.args.dataset
        cond_cw = "cw"

        filename_cw = f"{dataset}_enhance_cw_{conf.args.seed}_dist{conf.args.tgt_train_dist}"

        file_path_cw = conf.args.opt['file_path'] + "_cw"

        if conf.args.wds_path is None:
            correct_wrong_data = load_cache(filename_cw, cond_cw, file_path_cw, transform=None)
        else:
            correct_wrong_data = None
        if not correct_wrong_data:
            for corruption_i, corruption in enumerate(self.corruption_list):

                since = time.time()

                if conf.args.wds_path is None:
                    dataset = conf.args.dataset
                    cond = corruption

                    filename = f"{dataset}_enhance_{conf.args.seed}_dist{conf.args.tgt_train_dist}"

                    file_path = conf.args.opt['file_path'] + "_target_train_set"

                    target_train_set = load_cache(filename, cond, file_path, transform=None)

                    if not target_train_set:

                        print('##############Target Data Loading...##############')
                        self.set_seed()  # reproducibility
                        target_data_loader, _ = data_loader_module.domain_data_loader(conf.args.dataset, corruption,
                                                                            conf.args.opt['file_path'],
                                                                            batch_size=conf.args.enhance_tta_batchsize,
                                                                            valid_split=0,
                                                                            test_split=0, is_src=False,
                                                                            num_source=conf.args.num_source)

                        features, cl_labels, do_labels = [], [], []
                        for b_i, (feat, cl, dl) in enumerate(target_data_loader['train']):
                        # must be loaded from dataloader, due to transform in the __getitem__()

                            features.append(feat.squeeze(0))
                            cl_labels.append(cl.squeeze())
                            do_labels.append(dl.squeeze())

                        feats = torch.stack(features, axis = 0)
                        cls = torch.stack(cl_labels)

                        target_train_set = (feats, cls)
                        save_cache(target_train_set, filename, cond, file_path, transform=None)

                        time_elapsed = time.time() - since
                        print('Data Loading Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


                    feats, cls = target_train_set

                    # randomly sampled from loaded data
                    num_budget_per_corruption = int(conf.args.ass_num * len(feats) / 64)
                    selected_indexes = random.sample(range(len(feats)), num_budget_per_corruption)
                    selected_active_feats = feats[selected_indexes]
                    selected_active_cls = cls[selected_indexes]
                    selected_active_dls = torch.tensor([corruption_i] * len(selected_indexes))

                else:
                    url = os.path.join(conf.args.wds_path, f"{conf.args.dataset}_{conf.args.seed}_dist1_{corruption}.tar")

                    preproc = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                        ]
                    )
                    dataset = (
                        wds.WebDataset(url, shardshuffle=True)
                        .shuffle(1000)
                        .decode("pil")
                        .to_tuple("input.jpg", "output.cls", "dls.cls")
                        .map_tuple(preproc, lambda x : x)
                    )
                    num_budget_per_corruption = int(conf.args.ass_num *  50000 / 64)
                    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=num_budget_per_corruption)
                    selected_active_feats, selected_active_cls = iter(dataloader).next()

                    # selected_active_feats = feats[selected_indexes]
                    # selected_active_cls = cls[selected_indexes]
                    selected_active_dls = torch.tensor([corruption_i] * num_budget_per_corruption)

                # ask binary label using pre-trained src model
                self.net.eval()
                selected_feats_ = selected_active_feats
                selected_labels_ = selected_active_cls
                negative_cls = []

                selected_dataset = torch.utils.data.TensorDataset(selected_feats_, )
                selected_data_loader = DataLoader(selected_dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=False)


                with torch.no_grad():
                    y_pred = []
                    for (selected_feat,) in selected_data_loader:
                        selected_feat = selected_feat.to(device)
                        y_pred_ = self.net(selected_feat).max(1, keepdim=False)[1].cpu()
                        # y_entropy = softmax_entropy(y_logit)
                        # y_pred_ = y_logit.max(1, keepdim=False)[1]
                        y_pred += [y_pred_]
                    y_pred = torch.concat(y_pred)

                mask_correct: torch.Tensor = y_pred == selected_labels_

                correct_active_feats += [selected_active_feats[mask_correct]]
                correct_active_cls += [y_pred[mask_correct]]
                correct_active_dls += [selected_active_dls[mask_correct]]


                wrong_active_feats += [selected_active_feats[~mask_correct]]
                wrong_active_cls += [y_pred[~mask_correct]]
                wrong_active_dls += [selected_active_dls[~mask_correct]]

            correct_active_feats = torch.cat(correct_active_feats, axis = 0)
            correct_active_cls = torch.cat(correct_active_cls, axis = 0)
            correct_active_dls = torch.cat(correct_active_dls)

            wrong_active_feats = torch.cat(wrong_active_feats, axis = 0)
            wrong_active_cls = torch.cat(wrong_active_cls, axis = 0)
            wrong_active_dls = torch.cat(wrong_active_dls)

            correct_wrong_data = (correct_active_feats, correct_active_cls, correct_active_dls, wrong_active_feats, wrong_active_cls, wrong_active_dls)

            save_cache(correct_wrong_data, filename_cw, cond_cw, file_path_cw, transform=None)


        correct_active_feats, correct_active_cls, correct_active_dls, wrong_active_feats, wrong_active_cls, wrong_active_dls = correct_wrong_data

        correct_dataset = torch.utils.data.TensorDataset(correct_active_feats, correct_active_cls, correct_active_dls)
        correct_data_loader = DataLoader(correct_dataset, batch_size=conf.args.enhance_tta_batchsize,
                                 shuffle=True, drop_last=False, pin_memory=False)

        wrong_dataset = torch.utils.data.TensorDataset(wrong_active_feats, wrong_active_cls, wrong_active_dls)
        wrong_data_loader = DataLoader(wrong_dataset, batch_size=conf.args.enhance_tta_batchsize,
                                 shuffle=True, drop_last=False, pin_memory=False)


        print("correct_samples :", len(correct_active_feats))
        print("wrong_samples :", len(wrong_active_feats))
        print("total_samples :", len(correct_active_feats) + len(wrong_active_feats))

        # self.net.train()
        for epoch in range(conf.args.enhance_tta_epoch):
            loss_log = 0.0
            for (correct_feats_, correct_labels_, correct_domains_), (wrong_feats_, wrong_labels_, wrong_domains_) in zip(correct_data_loader, wrong_data_loader):
                correct_loss = torch.tensor([0.0]).to(device)
                wrong_loss = torch.tensor([0.0]).to(device)

                # correct samples
                correct_feats_ = correct_feats_.to(device)
                correct_labels_ = correct_labels_.to(device)

                correct_outputs = self.net(correct_feats_)
                correct_loss = self.class_criterion(correct_outputs, correct_labels_)

                # wrong samples
                wrong_feats_ = wrong_feats_.to(device)
                wrong_labels_ = wrong_labels_.to(device)

                wrong_outputs = self.net(wrong_feats_)

                # filter_idx = (wrong_outputs > 1 / conf.args.opt['num_class'])
                # for i in range(len(wrong_outputs)):
                #     filter_idx[i][wrong_labels_[i]] = 0.0

                # new_wrong_output = wrong_outputs.clone().detach()
                # new_wrong_output[~filter_idx] = 0.0
                # new_wrong_output = F.normalize(new_wrong_output, p=1, dim=1)

                # wrong_loss = self.class_criterion(wrong_outputs, new_wrong_output)

                wrong_loss = complement_CrossEntropyLoss(wrong_outputs, wrong_labels_)


                loss = conf.args.w_final_loss_correct * correct_loss + conf.args.w_final_loss_wrong * wrong_loss

                if conf.args.wandb:
                    import wandb
                    wandb.log({
                        'loss_correct': correct_loss.item(),
                        'loss_wrong': wrong_loss.item(),
                        'loss_total': loss.item(),
                    })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_log += loss.item()

            print(f"training : epoch {epoch + 1}, loss : {loss_log}")

        self.enable_running_stats()
        enhance_path = conf.args.enhance_save_path + f'/{conf.args.dataset}/enhance_{conf.args.seed}'
        if not os.path.exists(enhance_path):
            oldumask = os.umask(0)
            os.makedirs(enhance_path, 0o777)
            os.umask(oldumask)
        self.save_checkpoint(None, None, None, f'{enhance_path}/cp_last.pth.tar')

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


    def set_seed(self):
        torch.manual_seed(conf.args.seed)
        np.random.seed(conf.args.seed)
        random.seed(conf.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def get_bitta_ssl_loss(self):
        assert conf.args.enable_bitta
        self.disable_running_stats()

        loss = 0.0


        correct_feats, correct_preds, _, _ = self.active_mem.get_correct_memory()
        wrong_feats, wrong_preds, wrong_gt_labels, _ = self.active_mem.get_wrong_memory()

        if correct_preds is not None and len(correct_preds) > 0:

            if len(correct_preds) == 1:
                if not conf.args.use_learned_stats:
                    correct_feats = correct_feats + correct_feats
                else:
                    self.net.eval()

            correct_logits = self.net(torch.stack(correct_feats).to(device)).softmax(1)[:len(correct_preds)]
            correct_preds = torch.tensor(correct_preds).to(device)
            correct_loss = F.cross_entropy(correct_logits, correct_preds.detach())
            loss += conf.args.w_final_loss_correct * correct_loss

        if wrong_preds is not None and len(wrong_preds) > 0:

            if len(wrong_preds) == 1:
                if not conf.args.use_learned_stats:
                    wrong_feats = wrong_feats + wrong_feats
                else:
                    self.net.eval()

            wrong_logits = self.net(torch.stack(wrong_feats).to(device)).softmax(1)[:len(wrong_preds)]
            wrong_preds = torch.tensor(wrong_preds).to(device)
            wrong_loss = complement_CrossEntropyLoss(wrong_logits, wrong_preds.detach())
            loss += conf.args.w_final_loss_wrong * wrong_loss

        self.enable_running_stats()
        return loss
