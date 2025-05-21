import random

import numpy as np
import torch

import conf
from utils.custom_exceptions import *
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
import math

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]  # feat, cls, domain
        self.capacity = capacity
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1 : ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], []]

class HUS:
    def __init__(self, capacity, threshold=None):
        self.data = [[[], [], [], [], []] for _ in
                     range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf, cls
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        self.threshold = threshold

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], [], [], []]
        for data_per_cls in data:
            feats, cls, dls, conf, gt_cls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)
            tmp_data[3].extend(conf)
            tmp_data[4].extend(gt_cls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 5)
        pseudo_cls = instance[1]
        self.counter[pseudo_cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(pseudo_cls)

        if is_add:
            for i, dim in enumerate(self.data[pseudo_cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][0])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][0])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[[], [], [], [], []] for _ in range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf, cls


class ConfFIFO:
    def __init__(self, capacity, threshold):
        self.data = [[], [], [], [], []] # feat, pseudo_cls, domain, conf, cls
        self.capacity = capacity
        self.threshold = threshold
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        self.threshold = state_dict['threshold']
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold
        return dic

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 5)

        if instance[3] < self.threshold:
            return

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], [], [], []]


class ReplayMemory:
    def __init__(self, batch_size, interval):
        self.batch_size = batch_size
        self.interval = interval
        self.features = None
        self.pseudo_labels = None
        self.confidences = None

    def get_memory(self):
        return self.features, self.pseudo_labels, self.confidences

    def pop_memory(self):
        target_size = [self.batch_size, len(self.features) - self.batch_size]
        feats, self.features = torch.split(self.features, target_size)
        pls, self.pseudo_labels = torch.split(self.pseudo_labels, target_size)
        confs, self.confidences = torch.split(self.confidences, target_size)
        return feats, pls, confs

    def add_instance(self, instance):
        """
        Assumes incoming features and pseudo_labels are in shape of (B, ...) and (B, N)
        """
        assert (len(instance) == 3)  # features, pseudo_labels
        self.features = torch.cat((self.features, instance[0])) if self.features is not None else instance[0]
        self.pseudo_labels = torch.cat((self.pseudo_labels, instance[1])) if self.pseudo_labels is not None else instance[1]
        self.confidences = torch.cat((self.confidences, instance[2])) if self.confidences is not None else instance[2]

    def reset(self):
        self.features = None
        self.pseudo_labels = None
        self.confidences = None


class Uniform:
    def __init__(self, capacity):
        self.data = [[[], [], [], [], []] for _ in
                     range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf, cls
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        self.threshold = 0.0

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 5)
        pseudo_cls = instance[1]
        self.counter[pseudo_cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(pseudo_cls)

        if is_add:
            for i, dim in enumerate(self.data[pseudo_cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][0])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][0])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[[], [], [], [], []] for _ in range(conf.args.opt['num_class'])]  # feat, pseudo_cls, domain, conf, cls

class PBRS():

    def __init__(self, capacity):
        self.data = [[[], [], []] for _ in range(conf.args.opt['num_class'])] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        pass

    def reset(self):
        self.data = [[[], [], []] for _ in range(conf.args.opt['num_class'])] #feat, pseudo_cls, domain, cls, loss

    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

class IncKMeans:
    def __init__(self, capacity):
        self.data = [[], [], [], [], [], []]  # feat, cls, domain, extracted_feats, weight, entropy
        self.high_buffer = [[], [], [], [], [], []]
        self.low_buffer = [[], [], [], [], [], []]
        self.capacity = capacity
        self.number_of_cluster = 10
        self.incremental_rate = conf.args.atta_inc_rate
        self.max_iter = 100000
        self.n_init = 20
        
        self.clustering_algo = KMeans(n_clusters=self.number_of_cluster, max_iter=self.max_iter, n_init = self.n_init)

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        self.high_buffer = [ls[:] for ls in state_dict['high_buffer']]
        self.low_buffer = [ls[:] for ls in state_dict['low_buffer']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        dic['high_buffer'] = [ls[:] for ls in self.high_buffer]
        dic['low_buffer'] = [ls[:] for ls in self.low_buffer]
        return dic

    def get_memory(self):
        feats, cls, dls, extracted_feats, weights, ents = self.data
        low_feats, low_cls, low_dls, low_extracted_feats, low_weights, low_ents = self.low_buffer
        res = (feats + low_feats,
               cls + low_cls,
               dls + low_dls,
               extracted_feats + low_extracted_feats,
               weights + low_weights,
               ents + low_ents
               )
        return res

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 6)
        raise NotImplementedError
        # for i, dim in enumerate(self.added_buffer):
        #     dim.append(instance[i])

        # if self.get_occupancy() >= self.capacity:
        #     raise custom_exceptions.OutOfBudgetError(self.capacity)
    
    def add_high_instance(self, instance):
        assert (len(instance) == 6)
        
        for i, dim in enumerate(self.high_buffer):
            dim.append(instance[i])
        
    def add_low_instance(self, instance):
        assert (len(instance) == 6)
        
        for i, dim in enumerate(self.low_buffer):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        
    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], [], [], [], []]
        self.high_buffer = [[], [], [], [], [], []]
        self.low_buffer = [[], [], [], [], [], []]
        
    def reset_high_buffer(self):
        self.high_buffer = [[], [], [], [], [], []]
        
    def clustering(self, weight_cluster_by_ent = False):
    
        prev_samples, prev_cls, prev_dls, prev_extracted_feats, prev_weights, prev_ents = self.data
        new_samples, new_cls, new_dls, new_extracted_feats, new_weights, new_ents = self.high_buffer
        
        if weight_cluster_by_ent:
            prev_ent_weights = np.array(prev_weights) * np.array(prev_ents)
            new_ent_weights = np.array(prev_weights) * np.array(prev_ents)

            res = self.clustering_algo.fit(X=prev_extracted_feats + new_extracted_feats,
                                        sample_weight=np.concatenate([prev_ent_weights, new_ent_weights]))
        else:
            res = self.clustering_algo.fit(X=prev_extracted_feats + new_extracted_feats,
                                        sample_weight=prev_weights + new_weights)
            
        prev_anchors_labels = res.labels_[:len(prev_samples)]
        new_samples_labels = res.labels_[len(prev_samples):]

        unique_prev_anchors_labels = set(prev_anchors_labels)  # set of existing anchors' labels
        unique_new_samples_labels = set(new_samples_labels)  # set of new samples' labels
        
        unique_new_anchors_labels = unique_new_samples_labels - unique_new_samples_labels.intersection(unique_prev_anchors_labels)
        unique_new_anchors_labels = list(unique_new_anchors_labels)  # unique set of new-samples-only clusters' labels
        unique_prev_anchors_labels = list(unique_prev_anchors_labels)  # unique set of existing-anchors-containing clusters' labels

        if len(prev_samples) + len(unique_new_anchors_labels) > self.capacity:  # more than limit : not update
            return
        
        weight_counter = {}

        for i in range(len(prev_samples)):
            if prev_anchors_labels[i] not in weight_counter:
                weight_counter[prev_anchors_labels[i]] = [0, 1]  # N_samples, N_anchors
            else:
                weight_counter[prev_anchors_labels[i]][1] += 1  # increase N_anchors
                
        for i in range(len(new_samples)):
            if new_samples_labels[i] not in weight_counter:
                weight_counter[new_samples_labels[i]] = [1, 1]  # N_samples, N_anchors
            else:
                weight_counter[new_samples_labels[i]][0] += 1  # increase N_samples
        
        for k, v in weight_counter.items():
            weight_counter[k] = v[0]/v[1]  # N_samples / N_anchors

        updated_samples, updated_cls, updated_dls, updated_extracted_feats, updated_weights, updated_ents = [], [], [], [], [], []
       
        # update old anchors
        updated_samples, updated_cls, updated_dls, updated_extracted_feats, updated_ents = prev_samples, prev_cls, prev_dls, prev_extracted_feats, prev_ents
        for i in range(len(prev_samples)):
            updated_weights += [prev_weights[i] + weight_counter[prev_anchors_labels[i]]]
        
        # add new anchors
        centroids = np.array(res.cluster_centers_)
        min_dict = {}
        for i in range(len(new_samples)):
            label = new_samples_labels[i]
            if label in unique_prev_anchors_labels:  # skip if existing-anchors-containing cluster
                continue
            
            dist = np.linalg.norm(new_extracted_feats[i] - centroids[label])  # l2-norm distance
            if label not in min_dict:
                min_dict[label] = (dist, i)  # (distance, index)
            else:
                if dist < min_dict[label][0]:
                    min_dict[label] = (dist, i)
                else:
                    pass
                
        for new_anchor in unique_new_anchors_labels:  # get anchors for new-samples-only clusters
            min_i = min_dict[new_anchor][1]
            updated_samples += [new_samples[min_i]]
            updated_cls += [new_cls[min_i]]
            updated_dls += [new_dls[min_i]]
            updated_extracted_feats += [new_extracted_feats[min_i]]
            updated_weights += [weight_counter[new_anchor]]
            updated_ents += [new_ents]
        
        self.data = [updated_samples, updated_cls, updated_dls, updated_extracted_feats, updated_weights, updated_ents]
        self.high_buffer = [[], [], [], [], [], []]
        
        self.number_of_cluster += self.incremental_rate
        self.clustering_algo = KMeans(n_clusters=self.number_of_cluster, max_iter=self.max_iter, n_init = self.n_init)
        
        if self.get_occupancy() > self.capacity:
            raise OutOfBudgetError(self.capacity)
    
    def get_memory_anchors(self):
        return self.data
        
    def set_memory_anchors(self,  feats, cls, dls, extracted_feats, weights, ents):
        self.data =  feats, cls, dls, extracted_feats, weights, ents
        return
    
    def get_balance_weight(self, cold_start = None):
        len_D_l = len(self.low_buffer[0])
        len_D_h = len(self.data[0])
        alpha = len_D_h/(len_D_l + len_D_h) # weight high
        nalpha = 1 - alpha # weight low
        if cold_start != None: # based on official implementation
            if len_D_l < cold_start:
                alpha = min(0.2, alpha)
                nalpha = 1 - alpha
        return nalpha, alpha

    def increase_capacity(self, n):
        self.capacity += n
    

class BinaryIncKMeans:
    def __init__(self):
        self.data = [[], [], [], [], [], [], [], [], []]  # feat, cls, domain, extracted_feats, weight, correct, gt, pred_cls, ents (for debugging)
        self.high_buffer =[[], [], [], [], [], [], [], [], []]
        self.low_buffer = [[], [], [], [], [], [], [], [], []]
        self.unlabeled_buffer = [[], [], [], [], [], [], [], [], []]
        self.capacity = 300
        self.number_of_cluster = 10
        self.incremental_rate = 1
        self.max_iter = 100000
        self.n_init = 20
        self.fixing_type = conf.args.fixing_type
        self.fixing_count = [0, 0] # number of total fixing, number of successfully fixing

        
        self.clustering_algo = KMeans(n_clusters=self.number_of_cluster, max_iter=self.max_iter, n_init = self.n_init)

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        self.high_buffer = [ls[:] for ls in state_dict['high_buffer']]
        self.low_buffer = [ls[:] for ls in state_dict['low_buffer']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        dic['high_buffer'] = [ls[:] for ls in self.high_buffer]
        dic['low_buffer'] = [ls[:] for ls in self.low_buffer]
        return dic

    def get_memory(self):
        feats, cls, dls, extracted_feats, weights, corrects, gts, pred_cls, ents = self.data
        low_feats, low_cls, low_dls, low_extracted_feats, low_weights, low_corrects, low_gts, low_pred_cls, low_ents = self.low_buffer
        return (feats + low_feats, 
               cls + low_cls,
               dls + low_dls, 
               extracted_feats + low_extracted_feats, 
               weights + low_weights, 
               corrects + low_corrects, 
               gts + low_gts,
               pred_cls + low_pred_cls,
               ents + low_ents)

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 9)
        raise NotImplementedError
    
    def add_high_instance(self, instance):
        assert (len(instance) == 9)
        
        for i, dim in enumerate(self.high_buffer):
            dim.append(instance[i])
        
    def add_low_instance(self, instance):
        assert (len(instance) == 9)
        
        for i, dim in enumerate(self.low_buffer):
            dim.append(instance[i])
        
    def add_unlabeled_instance(self, instance):
        assert (len(instance) == 9)

        
        for i, dim in enumerate(self.unlabeled_buffer):
            if len(dim) >= 64:
                dim = dim[-63:]
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        
    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], [], [], [], [], [], [], []]
        self.high_buffer = [[], [], [], [], [], [], [], [], []]
        self.low_buffer = [[], [], [], [], [], [], [], [], []]
        self.unlabeled_buffer = [[], [], [], [], [], [], [], [], []]
        
    def reset_high_buffer(self):
        self.high_buffer = [[], [], [], [], [], [], [], [], []]
    
    def reset_unlabeled_buffer(self):
        self.unlabeled_buffer = [[], [], [], [], [], [], [], [], []]
        
    def clustering(self):
    
        prev_samples, prev_cls, prev_dls, prev_extracted_feats, prev_weights, prev_corrects, prev_gts, prev_pred_cls, prev_ents = self.data
        new_samples, new_cls, new_dls, new_extracted_feats, new_weights, new_corrects, new_gts, new_pred_cls, new_ents = self.high_buffer
        res = self.clustering_algo.fit(X=prev_extracted_feats + new_extracted_feats,
                                       sample_weight=prev_weights + new_weights)

        prev_anchors_labels = res.labels_[:len(prev_samples)]
        new_samples_labels = res.labels_[len(prev_samples):]

        unique_prev_anchors_labels = set(prev_anchors_labels)  # set of existing anchors' labels
        unique_new_samples_labels = set(new_samples_labels)  # set of new samples' labels
        
        unique_new_anchors_labels = unique_new_samples_labels - unique_new_samples_labels.intersection(unique_prev_anchors_labels)
        unique_new_anchors_labels = list(unique_new_anchors_labels)  # unique set of new-samples-only clusters' labels
        unique_prev_anchors_labels = list(unique_prev_anchors_labels)  # unique set of existing-anchors-containing clusters' labels

        if len(prev_samples) + len(unique_new_anchors_labels) > self.capacity:  # more than limit : not update
            return
        
        weight_counter = {} # key = cluster, value = N_samples, N_anchors
        correct_anchors = {} # key = cluster, value = anchors that correct
        wrong_anchors = {} # key = cluster, value = anchors that wrong

        for i in range(len(prev_samples)):
            if prev_anchors_labels[i] not in weight_counter:
                weight_counter[prev_anchors_labels[i]] = [0, 1]  # N_samples, N_anchors
                if prev_corrects[i] == 1: # correct
                    correct_anchors[prev_anchors_labels[i]] = [(prev_extracted_feats[i], prev_cls[i], i)]
                    wrong_anchors[prev_anchors_labels[i]] = []
                elif prev_corrects[i] in [0, 2]: # wrong or used to be wrong
                    correct_anchors[prev_anchors_labels[i]] = []
                    wrong_anchors[prev_anchors_labels[i]] = [(prev_extracted_feats[i], prev_cls[i], i)]
                else:
                    raise ValueError
            else:
                weight_counter[prev_anchors_labels[i]][1] += 1  # increase N_anchors  
                if prev_corrects[i] == 1: # correct
                    correct_anchors[prev_anchors_labels[i]] += [(prev_extracted_feats[i], prev_cls[i], i)]
                elif prev_corrects[i] in [0, 2]: # wrong or used to be wrong
                    wrong_anchors[prev_anchors_labels[i]] += [(prev_extracted_feats[i], prev_cls[i], i)]
                else:
                    raise ValueError      
        
        
        # fixing the wrong anchor with different method (self.fixing_type)
        if self.fixing_type != None:
            for k in wrong_anchors.keys(): # for each cluster that contains wrong anchor
                for wrong_anchor in wrong_anchors[k]: # for each wrong anchor in that cluster
                    wrong_extracted_feat, wrong_cls, wrong_i = wrong_anchor
                    
                    if self.fixing_type == "min":
                        # fix the label of wrong anchor to the nearest correct anchor in the same cluster
                        min_dist_rec = (float("inf"), None) # dist, new_correct_cls
                        for correct_anchor in correct_anchors[k]:
                            correct_extracted_feat, correct_cls, correct_i = correct_anchor
                            dist = np.linalg.norm(wrong_extracted_feat - correct_extracted_feat)
                            if min_dist_rec[0] > dist:
                                min_dist_rec = ((dist, correct_cls))
                        if min_dist_rec[1] != None:
                            prev_cls[wrong_i] = correct_cls # correct the wrong label
                            assert prev_corrects[wrong_i] in [0, 2]
                            prev_corrects[wrong_i] = 2
                            
                            # for logging
                            self.fixing_count[0] += 1 # total fixing
                            self.fixing_count[1] += (1 if prev_cls[wrong_i] == prev_gts[wrong_i] else 0) # successfully fixing
                     
                    elif self.fixing_type == "majority":
                        # fix the label of wrong anchor to the most popular correct anchor in the same cluster
                        correct_anchor_counter = {}
                        for correct_anchor in correct_anchors[k]:
                            correct_extracted_feat, correct_cls, correct_i = correct_anchor
                            if correct_cls not in correct_anchor_counter:
                                correct_anchor_counter[correct_cls] = 1
                            else:
                                correct_anchor_counter[correct_cls] += 1
                        max_count_anchor = [-float('inf'), None] # count, correct_cls
                        for k_, v_ in correct_anchor_counter.items():
                            if v_ > max_count_anchor[0]:
                                max_count_anchor = [v_, k_]
                        
                        if max_count_anchor[1] != None:
                            prev_cls[wrong_i] = max_count_anchor[1]
                            assert prev_corrects[wrong_i] in [0, 2]
                            prev_corrects[wrong_i] = 2
                            
                            # for logging
                            self.fixing_count[0] += 1 # total fixing
                            self.fixing_count[1] += (1 if prev_cls[wrong_i] == prev_gts[wrong_i] else 0) # successfully fixing
                    
                    elif self.fixing_type == "ideal":
                        if len(correct_anchors[k]) != 0:
                            prev_cls[wrong_i] = prev_gts[wrong_i] # iedally fix to correct class
                            assert prev_corrects[wrong_i] in [0, 2]
                            prev_corrects[wrong_i] = 2
                            
                            # for logging
                            self.fixing_count[0] += 1 # total fixing
                            self.fixing_count[1] += (1 if prev_cls[wrong_i] == prev_gts[wrong_i] else 0) # successfully fixing
                    
                    elif self.fixing_type == "selective_majority":
                        # fix the label of wrong anchor to the most popular correct anchor in the same cluster
                        correct_anchor_counter = {}
                        for correct_anchor in correct_anchors[k]:
                            correct_extracted_feat, correct_cls, correct_i = correct_anchor
                            if correct_cls not in correct_anchor_counter:
                                correct_anchor_counter[correct_cls] = 1
                            else:
                                correct_anchor_counter[correct_cls] += 1
                        max_count_anchor = [-float('inf'), None] # count, correct_cls
                        total_count_anchor = 0
                        for k_, v_ in correct_anchor_counter.items():
                            total_count_anchor += v_
                            if v_ > max_count_anchor[0]:
                                max_count_anchor = [v_, k_]
                        
                        if max_count_anchor[1] != None:
                            percent_majority = max_count_anchor[0]/total_count_anchor
                            if total_count_anchor >= 2 and percent_majority >= 0.7 and max_count_anchor[1] != prev_cls[wrong_i]:
                                prev_cls[wrong_i] = max_count_anchor[1]
                                assert prev_corrects[wrong_i] in [0, 2]
                                prev_corrects[wrong_i] = 2
                                
                                # for logging
                                self.fixing_count[0] += 1 # total fixing
                                self.fixing_count[1] += (1 if prev_cls[wrong_i] == prev_gts[wrong_i] else 0) # successfully fixing
                    
                    elif self.fixing_type == "new_pred_cls":
                        if prev_pred_cls[wrong_i] != prev_cls[wrong_i] and prev_ents[wrong_i] > 0.9:
                            # prev_cls[wrong_i] = prev_pred_cls[wrong_i]
                            # assert prev_corrects[wrong_i] in [0, 2]
                            # prev_corrects[wrong_i] = 2
                            
                            # for logging
                            self.fixing_count[0] += 1 # total fixing
                            self.fixing_count[1] += (1 if prev_pred_cls[wrong_i] == prev_gts[wrong_i] else 0) # successfully fixing
                    else:
                        raise ValueError(self.fixing_type)
        
        for i in range(len(new_samples)):
            if new_samples_labels[i] not in weight_counter:
                weight_counter[new_samples_labels[i]] = [1, 1]  # N_samples, N_anchors
            else:
                weight_counter[new_samples_labels[i]][0] += 1  # increase N_samples
        
        for k, v in weight_counter.items():
            weight_counter[k] = v[0]/v[1]  # N_samples / N_anchors

        updated_samples, updated_cls, updated_dls, updated_extracted_feats, updated_weights, updated_corrects, updated_gts, updated_pred_cls, updated_ents = [], [], [], [], [], [], [], [], []
       
        # update old anchors
        updated_samples, updated_cls, updated_dls, updated_extracted_feats, updated_corrects, updated_gts, updated_pred_cls, updated_ents = prev_samples, prev_cls, prev_dls, prev_extracted_feats, prev_corrects, prev_gts, prev_pred_cls, prev_ents
        for i in range(len(prev_samples)):
            updated_weights += [prev_weights[i] + weight_counter[prev_anchors_labels[i]]]
        
        # add new anchors
        centroids = np.array(res.cluster_centers_)
        min_dict = {}
        for i in range(len(new_samples)):
            label = new_samples_labels[i]
            if label in unique_prev_anchors_labels:  # skip if existing-anchors-containing cluster
                continue
            
            dist = np.linalg.norm(new_extracted_feats[i] - centroids[label])  # l2-norm distance
            if label not in min_dict:
                min_dict[label] = (dist, i)  # (distance, index)
            else:
                if dist < min_dict[label][0]:
                    min_dict[label] = (dist, i)
                else:
                    pass
                
        for new_anchor in unique_new_anchors_labels:  # get anchors for new-samples-only clusters
            min_i = min_dict[new_anchor][1]
            updated_samples += [new_samples[min_i]]
            updated_cls += [new_cls[min_i]]
            updated_dls += [new_dls[min_i]]
            updated_extracted_feats += [new_extracted_feats[min_i]]
            updated_corrects += [new_corrects[min_i]]
            updated_gts += [new_gts[min_i]]
            updated_pred_cls += [new_pred_cls[min_i]]
            updated_ents += [new_ents[min_i]]
            updated_weights += [weight_counter[new_anchor]]
            
        self.data = [updated_samples, 
                     updated_cls, 
                     updated_dls, 
                     updated_extracted_feats, 
                     updated_weights, 
                     updated_corrects, 
                     updated_gts,
                     updated_pred_cls,
                     updated_ents]
        
        self.high_buffer = [[], [], [], [], [], [], [], [], []]
        
        self.number_of_cluster += self.incremental_rate
        self.clustering_algo = KMeans(n_clusters=self.number_of_cluster, max_iter=self.max_iter, n_init = self.n_init)
        
        if self.get_occupancy() > self.capacity:
            raise OutOfBudgetError(self.capacity)
    
    def get_memory_anchors(self):
        return self.data
        
    def set_memory_anchors(self,  feats, cls, dls, extracted_feats, weights, corrects, gts, pred_cls ,ents):
        self.data =  feats, cls, dls, extracted_feats, weights, corrects, gts, pred_cls, ents
        return
    
    def get_balance_weight(self):
        len_D_l = len(self.low_buffer[0])
        len_D_h = len(self.data[0])
        return len_D_l/(len_D_l + len_D_h) , len_D_h/(len_D_l + len_D_h)

    def get_unlabeled_samples(self):
        return self.unlabeled_buffer

class EADA:
    def __init__(self, capacity):
        self.data = [[], [], []]  # feat, cls, domain # unlabeled data (FIFO)
        self.labeled_target_buffer = [[], [], [], []] # feat, cls, domain, correct
        self.src_like_buffer = [[], [], []] # feat, cls, domain
        self.capacity = capacity
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        return self.data

    def get_src_like(self):
        return self.src_like_buffer

    def get_labeled_target(self):
        return self.labeled_target_buffer

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])
    
    def add_low_instance(self, instance):
        assert (len(instance) == 3)
        
        for i, dim in enumerate(self.src_like_buffer):
            dim.append(instance[i])
            
    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1, ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], []]
    
    def selection(self, active_ratio, totality, model): # select data from self.data (unlabeled pool) to self.labeled_target_buffer
        model.eval()
        first_stat = list()
        with torch.no_grad():
            tgt_img, tgt_lbl, _ = self.data
            tgt_img, tgt_lbl = torch.stack(tgt_img).to(device), torch.tensor(tgt_lbl).to(device)

            tgt_out = model(tgt_img)
            pred_cls = tgt_out.max(1, keepdim=False)[0]
            # MvSM of each sample
            # minimal energy - second minimal energy
            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            # free energy of each sample
            output_div_t = -1.0 * tgt_out / conf.args.energy_beta
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * conf.args.energy_beta * output_logsumexp

            for i in range(len(free_energy)):
                first_stat.append([i, tgt_lbl[i].item(), mvsm_uncertainty[i].item(), free_energy[i].item()])

        first_sample_ratio = conf.args.first_sample_ratio
        first_sample_num = math.ceil(totality * first_sample_ratio)
        second_sample_ratio = active_ratio / conf.args.first_sample_ratio
        second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

        # the first sample using \mathca{F}, higher value, higher consideration # free_energy
        first_stat = sorted(first_stat, key=lambda x: x[-1], reverse=True)
        second_stat = first_stat[:first_sample_num]

        # the second sample using \mathca{U}, higher value, higher consideration # mvsm_uncertainty
        second_stat = sorted(second_stat, key=lambda x: x[-2], reverse=True)
        second_stat = np.array(second_stat)

        active_samples = second_stat[:second_sample_num]
        selected_index = set(active_samples[:, 0].astype(int))

        new_data = [[],[],[]]
        for i in range(len(self.data[0])):
            if i not in selected_index: #not selected
                for dim_i, dim in enumerate(new_data):
                    dim.append(self.data[dim_i][i])
            else: # selected
                for dim_i, dim in enumerate(self.labeled_target_buffer):
                    if dim_i != 3:
                        dim.append(self.data[dim_i][i])
                    else:
                        if pred_cls[i] == self.data[1][i]:
                            correct = 1
                        else:
                            correct = 0
                        dim.append(correct)
        
        self.data = new_data
    
        return active_samples


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.label = label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age, self.label

    def empty(self):
        return self.data == "empty"


class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def reset(self):
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def set_memory(self, state_dict):  # for tta_attack
        self.capacity = state_dict['capacity']
        self.num_class = state_dict['num_class']
        self.per_class = state_dict['per_class']
        self.lambda_t = state_dict['lambda_t']
        self.lambda_u = state_dict['lambda_u']
        self.data = [ls[:] for ls in state_dict['data']]

    def save_state_dict(self):
        dic = {}
        dic['capacity'] = self.capacity
        dic['num_class'] = self.num_class
        dic['per_class'] = self.per_class
        dic['lambda_t'] = self.lambda_t
        dic['lambda_u'] = self.lambda_u
        dic['data'] = [ls[:] for ls in self.data]

        return dic

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance):
        assert (len(instance) == 4)
        x, prediction, uncertainty, label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, label=label)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if all_occupancy < self.capacity:
            return True
        if class_occupied < self.per_class:
            majority_classes = self.get_majority_classes()
            return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: list[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(
            self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []
        tmp_label = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)
                tmp_label.append(item.label)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age, tmp_label

