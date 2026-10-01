import torch.nn
from torch import optim

import conf
from utils.memory import FIFO
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
import copy
from utils.atta_dataloader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset
from munch import Munch
from joblib import parallel_backend
from sklearn.metrics import pairwise_distances_argmin_min
from utils.atta_metric import Metric
import numpy as np
import random

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SimATTA(DNN):
    def __init__(self, *args, **kwargs):
        # self.atta_src_net = None
        super(SimATTA, self).__init__(*args, **kwargs)
        assert(conf.args.memory_type == "FIFO")
        
    def init_learner(self):
        
        self.atta_variables = {}
        self.atta_variables['freeze_bn'] = True
        
        for param in self.net.parameters():  #turn on requires_grad for all
            param.requires_grad = True

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats: # use learn stats
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        
        if conf.args.model == "resnet50_domainnet":
            self.atta_variables['teacher'] = self.deepcopy_resnet50_domainnet(self.net.to('cpu'))
        else:
            self.atta_variables['teacher'] = copy.deepcopy(self.net.to('cpu'))

        self.atta_variables['teacher'].eval()
        
        # change train mode
        self.net.original_train = self.net.train
        self.net.train = self.train_mode_freeze_bn
        self.freeze_bn()
        
        self.net.to(device)
        self.atta_variables['teacher'].to(device)
        self.update_teacher(0)  # copy student to teacher

        optimizer = optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'], momentum=0.9)

        #####-----######

        self.atta_variables['budgets'] = 0
        self.atta_variables['anchors'] = None
        self.atta_variables['source_anchors'] = None
        self.atta_variables['buffer'] = []
        self.atta_variables['n_clusters'] = 10
        self.atta_variables['nc_increase'] = conf.args.atta_inc_rate
        self.atta_variables['source_n_clusters'] = 100

        self.atta_variables['cold_start'] = conf.args.atta_cold_start

        self.atta_variables['consistency_weight'] = 0
        self.atta_variables['alpha_teacher'] = 0
        self.atta_variables['accumulate_weight'] = True
        self.atta_variables['weighted_entropy'] = 'both' # Union[Literal['low', 'high', 'both'], None]
        self.atta_variables['aggressive'] = True
        self.atta_variables['beta'] = 0
        self.atta_variables['alpha'] = 0.2

        # self.target_cluster = True if self.config.atta.SimATTA.target_cluster else False
        self.atta_variables['target_cluster'] = True
        # self.LE = True if self.config.atta.SimATTA.LE else False
        self.atta_variables['LE'] = True
        self.atta_variables['vis_round'] = 0
                
        
        if conf.args.dataset == "vlcs":
            self.atta_variables['eh'] = 1e-2
            self.atta_variables['el'] = 1e-3
            self.atta_variables['steps'] = 100
            self.atta_variables['train_bs'] = 16
            self.atta_variables['gpu_clustering'] = False
        elif conf.args.dataset == "pacs":
            self.atta_variables['eh'] = 1e-2
            self.atta_variables['el'] = 1e-4 
            self.atta_variables['steps'] = 30
            self.atta_variables['train_bs'] = 16
            self.atta_variables['gpu_clustering'] = False
        elif conf.args.dataset == "tiny-imagenet":
            self.atta_variables['eh'] = 1e-1
            self.atta_variables['el'] = 1e-1
            self.atta_variables['steps'] = 10
            self.atta_variables['train_bs'] = 32
            self.atta_variables['gpu_clustering'] = True
        elif conf.args.dataset in ["ccc", "imagenet"]:
            self.atta_variables['eh'] = 1e-1
            self.atta_variables['el'] = 1e-1
            self.atta_variables['steps'] = 10
            self.atta_variables['train_bs'] = 32
            self.atta_variables['gpu_clustering'] = False
        else:
            self.atta_variables['eh'] = conf.args.atta_upper_th
            self.atta_variables['el'] = conf.args.atta_lower_th
            self.atta_variables['steps'] = 30
            self.atta_variables['train_bs'] = conf.args.atta_batch_size
            self.atta_variables['gpu_clustering'] = False
            
        self.atta_variables['num_workers'] = 0 # 8
        self.atta_variables['stop_tol'] = 5
        metric = Metric()
        self.atta_variables['loss_func'] = metric.cross_entropy_with_logit

        self.atta_variables['count_budget'] = 0
        self.atta_variables['count_src_like'] = 0
        
        self.atta_variables['current_batch_budget'] = self.atta_variables['n_clusters'] -  conf.args.atta_limit_batch_active_sample if conf.args.atta_limit_batch_active_sample is not None else None
        
        return optimizer

    def test_time_adaptation(self):
    
        assert isinstance(self.mem, FIFO)
        feats, labels, _ = self.mem.get_memory()
        
        feats = torch.stack(feats).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)
    

        if conf.args.wandb:
            import wandb
            wandb.log({
                'num_batch_adapt': self.num_batch_adapt,
                'budget': self.atta_variables['count_budget'],
                'budget+prev':  self.atta_variables['count_budget'] +  self.atta_variables['current_batch_budget'] if self.atta_variables['count_budget'] is not None else self.atta_variables['count_budget'],
            })
        
        budget_limit = conf.args.atta_budget if conf.args.atta_budget != -1 else float('inf')
        
        if conf.args.memory_size != 1:
            
            if self.count_skip == conf.args.enable_skip:
                self.count_skip = 0

                if self.atta_variables['count_budget'] < budget_limit: # normal active sample selection
                ########### end of added code ############
                    outputs, closest, self.atta_variables['anchors'] = self.sample_select(self.net, feats, labels, 
                                                                                        self.atta_variables['anchors'], 
                                                                                        int(self.atta_variables['n_clusters']), 
                                                                                        1, ent_bound=self.atta_variables['eh'], 
                                                                                        incremental_cluster=self.atta_variables['target_cluster'])
                else:
                    outputs, closest = [], []
                    
            else:
                self.count_skip += 1
                closest = []
                
                
            if self.atta_variables['LE']: # src-like data selection
                _, _, self.atta_variables['source_anchors'] = self.sample_select(self.atta_variables['teacher'], 
                                                                                feats, labels, self.atta_variables['source_anchors'], 
                                                                                self.atta_variables['source_n_clusters'], 0, 
                                                                                use_pseudo_label=True, ent_bound=self.atta_variables['el'], 
                                                                                incremental_cluster=False)
            else:
                self.atta_variables['source_anchors'] = self.update_anchors(None, torch.tensor([]), None, None, None)
                
        else:
            
            self.count_bs1 += 1
            self.count_bs1 %= 64

            self.atta_variables['source_anchors'] = self.update_anchors(None, torch.tensor([]), None, None, None)

            if self.count_bs1 == 0:
                self.random_indexes_bs1 = np.arange(64)
                np.random.shuffle(self.random_indexes_bs1)
                self.random_indexes_bs1 = self.random_indexes_bs1[:3]
            
            if self.count_bs1 in self.random_indexes_bs1:
                
                with torch.no_grad():
                    outputs = self.net(feats)

                    pseudo_label = outputs.argmax(1).cpu().detach()
                    feats = feats.cpu().detach()
                    labels = labels.cpu().detach()
                    binary = pseudo_label == labels
            
                    self.atta_variables['anchors'] = self.update_anchors(None, feats, labels, None, 0.0)
                
                self.atta_variables['count_budget'] += 1
            closest = []
                
        if not self.atta_variables['target_cluster']:
            self.atta_variables['n_clusters'] = 0
        self.atta_variables['source_n_clusters'] = 100

        self.atta_variables['budgets'] += len(closest)
        if self.count_skip == 0:
            self.atta_variables['n_clusters'] += self.atta_variables['nc_increase']
        self.atta_variables['source_n_clusters'] += 1

        if conf.args.memory_size != 1:
            
            if conf.args.atta_limit_mem_size is not None:
                self.atta_variables['anchors'].data = self.atta_variables['anchors'].data[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].target = self.atta_variables['anchors'].target[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].feats = self.atta_variables['anchors'].feats[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].weight = self.atta_variables['anchors'].weight[-conf.args.atta_limit_mem_size:]
            
                self.atta_variables['source_anchors'].data = self.atta_variables['source_anchors'].data[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].target = self.atta_variables['source_anchors'].target[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].feats = self.atta_variables['source_anchors'].feats[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].weight = self.atta_variables['source_anchors'].weight[-conf.args.atta_limit_mem_size:]

                assert self.atta_variables['anchors'].num_elem() <= conf.args.atta_limit_mem_size
                assert self.atta_variables['source_anchors'].num_elem() <= conf.args.atta_limit_mem_size
                
                if self.atta_variables['anchors'].num_elem() == conf.args.atta_limit_mem_size:
                    self.atta_variables['n_clusters'] = self.atta_variables['anchors'].num_elem() + self.atta_variables['nc_increase']
                
            
            print(self.atta_variables['anchors'].num_elem(), self.atta_variables['source_anchors'].num_elem(), 
                self.atta_variables['count_budget'], self.atta_variables['count_src_like'])
                
            if self.atta_variables['source_anchors'].num_elem() > 0:
                
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['source_anchors'])
            else:
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['anchors'])
            
            ## added code to contraint about budget ##
            budget_limit = conf.args.atta_budget if conf.args.atta_budget != -1 else float('inf')
            if self.atta_variables['anchors'].num_elem() < budget_limit:
            ########### end of added code ############
                self.atta_variables['anchors'] = self.update_anchors_feats(self.atta_variables['anchors']) 

        else:
            if self.atta_variables['anchors'] is not None:
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['anchors'])
        
    #  not used
    @torch.no_grad()
    def val_anchor(self, loader):
        self.net.eval()
        val_loss = 0
        val_acc = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # output = self.fc(self.encoder(data))
            output = self.net(data)
            val_loss += self.atta_variables['loss_func'](output, target, reduction='sum').item()
            val_acc += self.atta_variables['loss_func'](target, output) * len(data)
        val_loss /= len(loader.sampler)
        val_acc /= len(loader.sampler)
        return val_loss, val_acc

    def update_teacher(self, alpha_teacher):  # , iteration):
        for t_param, s_param in zip(self.atta_variables['teacher'].parameters(), self.net.parameters()):
            t_param.data[:] = alpha_teacher * t_param[:].data[:] + (1 - alpha_teacher) * s_param[:].data[:]
        if not self.atta_variables['freeze_bn']:
            for tm, m in zip(self.atta_variables['teacher'].modules(), self.net.modules()):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    tm.running_mean = alpha_teacher * tm.running_mean + (1 - alpha_teacher) * m.running_mean
                    tm.running_var = alpha_teacher * tm.running_var + (1 - alpha_teacher) * m.running_var


    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        self.net.train()

        # torch.manual_seed(0)
        source_loader = InfiniteDataLoader(TensorDataset(source_anchors.data, source_anchors.target), weights=None,
                                           batch_size=self.atta_variables['train_bs'],
                                           num_workers=self.atta_variables['num_workers'])
        # torch.manual_seed(0)
        target_loader = InfiniteDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
                                             batch_size=self.atta_variables['train_bs'], num_workers=self.atta_variables['num_workers'])
        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_anchors.num_elem())
        
        # if src-like too small -> set contribution of src like to 0.2
        if source_anchors.num_elem() < self.atta_variables['cold_start']: 
            alpha = min(0.2, alpha)

        ST_loader = iter(zip(source_loader, target_loader))
        val_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
                                    batch_size=self.atta_variables['train_bs'], num_workers=self.atta_variables['num_workers'])
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'], momentum=0.9)

        # cluster train #
        delay_break = False
        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        for i, ((S_data, S_targets), (T_data, T_targets)) in enumerate(ST_loader):
            S_data, S_targets = S_data.to(device), S_targets.to(device)
            T_data, T_targets = T_data.to(device), T_targets.to(device)
            L_T = self.one_step_train(S_data, S_targets, T_data, T_targets, alpha, self.optimizer)
            # self.update_teacher(self.atta_variables['alpha_teacher'])
            if len(loss_window) < self.atta_variables['stop_tol']:
                loss_window.append(L_T.item())
            else:
                mean_loss = np.mean(loss_window)
                tol += 1
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    tol = 0
                if tol > 5:
                    break
                loss_window = []

            if i > self.atta_variables['steps']:
                break

    def one_step_train(self, S_data, S_targets, T_data, T_targets, alpha, optimizer):
        #one step train
        # torch.manual_seed(0)
        S_out = self.net(S_data)
        T_out = self.net(T_data)
        # print(sum([i.sum() for i in list(self.net[1].parameters())]))
        L_S = self.atta_variables['loss_func'](S_out, S_targets)
        L_T = self.atta_variables['loss_func'](T_out, T_targets)
        loss = (1 - alpha) * L_S + alpha * L_T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(sum([i.sum() for i in list(self.net[1].parameters())]))
        return L_T

    def softmax_entropy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if y is None:
            if x.shape[1] == 1:
                x = torch.cat([x, -x], dim=1)
            return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        else:
            return - 0.5 * (x.softmax(1) * y.log_softmax(1)).sum(1) - 0.5 * (y.softmax(1) * x.log_softmax(1)).sum(1)
        
    
    def update_anchors(self, anchors, data, target, feats, weight):
        if anchors is None:
            anchors = Munch()
            anchors.data = data
            anchors.target = target
            anchors.feats = feats
            anchors.weight = weight
            anchors.num_elem = lambda: len(anchors.data)
        else:
            anchors.data = torch.cat([anchors.data, data])
            anchors.target = torch.cat([anchors.target, target])
            anchors.feats = torch.cat([anchors.feats, feats])
            anchors.weight = torch.cat([anchors.weight, weight])
        return anchors

    def update_anchors_feats(self, anchors):
        # sequential_data = torch.arange(200)[:, None]
        anchors_loader = FastDataLoader(TensorDataset(anchors.data), weights=None,
                                        batch_size=32, num_workers=self.atta_variables['num_workers'], sequential=True)

        anchors.feats = None
        self.net.eval()
        for data in anchors_loader:
            data = data[0].to(device)
            if anchors.feats is None:
               
                _, feats = self.net[1](self.net[0](data), get_embedding=True)
                
                anchors.feats = feats.cpu().detach()
            else:
                _, feats = self.net[1](self.net[0](data), get_embedding=True)
                # feats = self.net[0](data) 
                anchors.feats = torch.cat([anchors.feats, feats.cpu().detach()])
        return anchors
    
    
    @torch.no_grad()
    def sample_select(self, model, data, target, anchors, n_clusters, ent_beta, use_pseudo_label=False, ent_bound=1e-2, incremental_cluster=False):
        model.eval()
        
        
        outputs, feats = model[1](model[0](data), get_embedding=True)

        # feats = model[0](data) 
        # outputs = model[1](feats)
        
        pseudo_label = outputs.argmax(1).cpu().detach()
        data = data.cpu().detach()
        feats = feats.cpu().detach()
        target = target.cpu().detach()
        entropy = self.softmax_entropy(outputs).cpu()

        if not incremental_cluster:
            entropy = entropy.numpy()
            if ent_beta == 0:
                closest = np.argsort(entropy)[: n_clusters]
                closest = closest[entropy[closest] < ent_bound]
            elif ent_beta == 1:
                closest = np.argsort(entropy)[- n_clusters:]
                closest = closest[entropy[closest] >= ent_bound]
            else:
                raise NotImplementedError
            weights = torch.zeros(len(closest), dtype=torch.float)
        else:
            if ent_beta == 0:
                sample_choice = entropy < ent_bound
            elif ent_beta == 1:
                sample_choice = entropy >= ent_bound
            else:
                raise NotImplementedError

            data = data[sample_choice]
            target = target[sample_choice]
            feats = feats[sample_choice]
            pseudo_label = pseudo_label[sample_choice]

            if anchors:
                feats4cluster = torch.cat([anchors.feats, feats])
                sample_weight = torch.cat([anchors.weight, torch.ones(len(feats), dtype=torch.float)])

            else:
                feats4cluster = feats
                sample_weight = torch.ones(len(feats), dtype=torch.float)
                
            # defensive programming
            if len(feats4cluster) < n_clusters:
                raise ValueError('Too large eh : total number of candidate samples are less than n_clusters')
            
            if self.atta_variables['gpu_clustering']:
                from utils.fast_pytorch_kmeans import KMeans
                from joblib import parallel_backend
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, device=device).fit(
                    feats4cluster.to(device),
                    sample_weight=sample_weight.to(device))
                with parallel_backend('threading', n_jobs=8):
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels_ = kmeans.labels_
                
            else:
                from joblib import parallel_backend
                from sklearn.cluster import KMeans
                with parallel_backend('threading', n_jobs=8):
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, algorithm='elkan', random_state=conf.args.seed).fit(feats4cluster,
                                                                                                  sample_weight=sample_weight)
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels_ = kmeans.labels_

            if anchors:
                num_anchors = anchors.num_elem()
                prev_anchor_cluster = torch.tensor(kmeans_labels_[:num_anchors], dtype=torch.long)

                if self.atta_variables['accumulate_weight']:
                    # previous anchor weight accumulation
                    # Average the weight of the previous anchor if sharing the same cluster
                    num_prev_anchors_per_cluster = prev_anchor_cluster.unique(return_counts=True)
                    num_prev_anchors_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_prev_anchors_per_cluster_dict[num_prev_anchors_per_cluster[0].long()] = \
                    num_prev_anchors_per_cluster[1]

                    num_newsample_per_cluster = torch.tensor(kmeans.labels_).unique(return_counts=True)
                    num_newsample_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_newsample_per_cluster_dict[num_newsample_per_cluster[0].long()] = num_newsample_per_cluster[1]
                    assert (num_prev_anchors_per_cluster_dict[prev_anchor_cluster] == 0).sum() == 0
                    # accumulate the weight of the previous anchor
                    anchors.weight = anchors.weight + num_newsample_per_cluster_dict[prev_anchor_cluster] / \
                                          num_prev_anchors_per_cluster_dict[prev_anchor_cluster].float()

                anchored_cluster_mask = torch.zeros(len(raw_closest), dtype=torch.bool).index_fill_(0,
                                                                                                    prev_anchor_cluster.unique().long(),
                                                                                                    True)
                new_cluster_mask = ~ anchored_cluster_mask

                closest = raw_closest[new_cluster_mask] - num_anchors
                if (closest < 0).sum() != 0:
                    # The cluster's closest sample may not belong to the cluster. It makes sense to eliminate them.
                    print('new_cluster_mask: ', new_cluster_mask)
                    new_cluster_mask = torch.where(new_cluster_mask)[0]
                    print('new_cluster_mask: ', new_cluster_mask)
                    print(closest)
                    print(closest >= 0)
                    new_cluster_mask = new_cluster_mask[closest >= 0]
                    closest = closest[closest >= 0]

                
                weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1][new_cluster_mask]
            else:
                num_anchors = 0
                closest = raw_closest
                weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1]
        
            
        if ent_beta == 1:
            if conf.args.atta_limit_batch_active_sample is not None:
                current_batch_budget = conf.args.atta_limit_batch_active_sample + self.atta_variables['current_batch_budget']
                if len(closest) < current_batch_budget:
                    self.atta_variables['current_batch_budget'] = current_batch_budget - len(closest)
                else:
                    self.atta_variables['current_batch_budget'] = 0
                perm_index = np.random.permutation(len(closest))
                closest = closest[perm_index[:current_batch_budget]]
                weights = weights[perm_index[:current_batch_budget]]
                
            self.atta_variables['count_budget'] += len(closest)
        elif ent_beta == 0:
            self.atta_variables['count_src_like'] += len(closest)
        if use_pseudo_label:
            anchors = self.update_anchors(anchors, data[closest], pseudo_label[closest], feats[closest], weights)
        else:
            anchors = self.update_anchors(anchors, data[closest], target[closest], feats[closest], weights)

        return outputs, closest, anchors
    
    def enable_bn(self, model):
        if not self.atta_variables['freeze_bn']:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = 0.1
                    

    # freeze bn on network when in train mode
    def train_mode_freeze_bn(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        self.net.original_train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        # if False:
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def deepcopy_resnet50_domainnet(self, net):
        from models import ResNet
        copied_net = ResNet.ResNet50_DOMAINNET().to(device)
        copied_net = torch.nn.Sequential(copy.deepcopy(net[0]), copied_net)
        copied_net.train()
        for param in copied_net.parameters():  #turn on requires_grad for all
            param.requires_grad = True
        for module in copied_net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                #use of batch stats in train and eval modes: https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.requires_grad_(True)
                
        copied_net.load_state_dict(copy.deepcopy(net.state_dict()))
        return copied_net