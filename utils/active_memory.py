import random

import numpy as np
import torch

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

class ActivePriorityFIFO:
    def __init__(self, capacity, pop="min", delay=0):
        # feat, cls, domain, entropy
        self.correct_mem = [[], [], [], []]  # for correct samples
        self.wrong_mem = [[], [], [], []]  # for wrong samples
        self.u_mem = [[], [], [], []]  # for unlabeled samples : wait to be labeled
        self.capacity = capacity
        self.pop = pop
        self.delay = delay

        self.delayed_queue = []  # Queue for delayed insertions
        self.unlabeled_count = 0 # Count unlabeled samples

    def set_memory(self, state_dict):  # for tta_attack
        self.correct_mem = [ls[:] for ls in state_dict['correct_mem']]
        self.wrong_mem = [ls[:] for ls in state_dict['wrong_mem']]
        self.u_mem = [ls[:] for ls in state_dict['u_mem']]

        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['correct_mem'] = [ls[:] for ls in self.correct_mem]
        dic['wrong_mem'] = [ls[:] for ls in self.wrong_mem]
        dic['u_mem'] = [ls[:] for ls in self.u_mem]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        dic = {'correct_mem': self.correct_mem,
               'wrong_mem': self.wrong_mem,
               'u_mem': self.u_mem,
               'capacity': self.capacity}
        return dic

    def get_correct_memory(self):
        return self.correct_mem

    def get_wrong_memory(self):
        return self.wrong_mem

    def get_u_memory(self):
        return self.u_mem

    def get_occupancy(self, mem):
        return len(mem[0])  # need to be checked

    def add_instance(self, instance):
        raise NotImplementedError

    def add_correct_instance(self, instance):
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('correct', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.correct_mem, instance)

    def add_wrong_instance(self, instance):
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('wrong', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.wrong_mem, instance)

    def add_u_instance(self, instance):
        assert len(instance) == 4
        self.unlabeled_count += 1
        self._insert_instance(self.u_mem, instance)

        # After every 64 unlabeled samples (one batch), check delayed queue
        if self.unlabeled_count % conf.args.update_every_x == 0:
            self._process_delayed_queue()

    def _insert_instance(self, mem, instance):
        if self.get_occupancy(mem) >= self.capacity:
            self.remove_instance(mem, pop=self.pop)
        for i, dim in enumerate(mem):
            dim.append(instance[i])

    def _process_delayed_queue(self):
        # Process delayed insertions if delay batches have passed
        ready_instances = []
        for idx, (mem_type, instance, unlabeled_start) in enumerate(self.delayed_queue):
            if (self.unlabeled_count - unlabeled_start) // conf.args.update_every_x >= self.delay:
                ready_instances.append(idx)
                if mem_type == 'correct':
                    self._insert_instance(self.correct_mem, instance)
                elif mem_type == 'wrong':
                    self._insert_instance(self.wrong_mem, instance)

        # Remove inserted instances from delayed queue
        for idx in reversed(ready_instances):
            self.delayed_queue.pop(idx)

    def remove_instance(self, mem, pop=None):
        if pop == "min":
            target_idx = np.argmin(mem[3])
        elif pop == "max":
            target_idx = np.argmax(mem[3])
        else:
            target_idx = 0
        self.remove_instance_by_index(mem, target_idx)

    def remove_instance_by_index(self, mem, index):
        for dim in mem:
            dim.pop(index)
        return

    def remove_u_instance_by_index(self, index):
        self.remove_instance_by_index(self.u_mem, index)

    def reset(self):
        self.u_mem = [[], [], [], []]
        self.correct_mem = [[], [], [], []]
        self.wrong_mem = [[], [], [], []]


class ActivePriorityPBRS:
    def __init__(self, capacity, pop="min"):
        # feat, cls, domain, entropy
        self.correct_mem = [[], [], [], []]  # for correct samples
        self.wrong_mem = [[], [], [], []]  # for wrong samples
        self.u_mem = [[[], [], [], []] for _ in range(conf.args.opt['num_class'])]  # for unlabeled samples : wait to be labeled
        self.capacity = capacity
        self.pop = pop
        
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
    
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.correct_mem = [ls[:] for ls in state_dict['correct_mem']]
        self.wrong_mem = [ls[:] for ls in state_dict['wrong_mem']]
        self.u_mem = [ls[:] for ls in state_dict['u_mem']]

        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['correct_mem'] = [ls[:] for ls in self.correct_mem]
        dic['wrong_mem'] = [ls[:] for ls in self.wrong_mem]
        dic['u_mem'] = [ls[:] for ls in self.u_mem]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        dic = {'correct_mem': self.correct_mem,
               'wrong_mem': self.wrong_mem,
               'u_mem': self.u_mem,
               'capacity': self.capacity}
        return dic

    def get_correct_memory(self):
        return self.correct_mem

    def get_wrong_memory(self):
        return self.wrong_mem

    def get_u_memory(self):
        merged_u_mem = [[],[],[],[]]
        for cls_i in range(len(self.u_mem)):
            for data_i in range(len(merged_u_mem)):
                merged_u_mem[data_i] += self.u_mem[cls_i][data_i]
        return merged_u_mem

    def get_occupancy(self, mem):
        return len(mem[0])  # need to be checked

    def add_instance(self, instance):
        raise NotImplementedError

    def add_correct_instance(self, instance):
        assert (len(instance) == 4)

        if self.get_occupancy(self.correct_mem) >= self.capacity:
            self.remove_instance(self.correct_mem, pop=self.pop)

        for i, dim in enumerate(self.correct_mem):
            dim.append(instance[i])

    def add_wrong_instance(self, instance):
        assert (len(instance) == 4)

        if self.get_occupancy(self.wrong_mem) >= self.capacity:
            self.remove_instance(self.wrong_mem, pop=self.pop)

        for i, dim in enumerate(self.wrong_mem):
            dim.append(instance[i])

    #### PBRS ########################################################################
    
    def add_u_instance(self, instance):
        # assert (len(instance) == 4)

        # if self.get_occupancy(self.u_mem) >= self.capacity:
        #     self.remove_instance(self.u_mem)

        # for i, dim in enumerate(self.u_mem):
        #     dim.append(instance[i])
        
        assert (len(instance) == 5)
        cls = instance[4]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy(self.u_mem) >= self.capacity:
            is_add = self.remove_instance_pbrs(self.u_mem, cls)

        if is_add:
            for i, dim in enumerate(self.u_mem[cls]):
                dim.append(instance[i])
    
    def remove_instance_pbrs(self, mem, cls):
        largest_indices = self.get_largest_indices(mem)
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(mem[largest][0]))  # target index to remove
            for dim in mem[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(mem[cls][0]))  # target index to remove
                for dim in mem[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
    
    def get_occupancy_per_class(self, mem):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(mem):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class
    
    def get_largest_indices(self, mem):

        occupancy_per_class = self.get_occupancy_per_class(mem)
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices
    
    ################################################################################
    
    
    def remove_instance(self, mem, pop=None):
        if pop == "min":
            target_idx = np.argmin(mem[3])
        elif pop == "max":
            target_idx = np.argmax(mem[3])
        else:
            target_idx = 0
        self.remove_instance_by_index(mem, target_idx)

    def remove_instance_by_index(self, mem, index):
        for dim in mem:
            dim.pop(index)
        return

    def remove_u_instance_by_index(self, index):
        # self.remove_instance_by_index(self.u_mem, index)
        count = 0
        for cls_i in range(len(self.u_mem)):
            count += len(self.u_mem[cls_i][0])
            if count > index:
                for dim in self.u_mem[cls_i]:
                    dim.pop(index - count + len(self.u_mem[cls_i][0]))
                return
        raise ValueError
            

    def reset(self):
        self.u_mem = [[], [], [], []]
        self.correct_mem = [[], [], [], []]
        self.wrong_mem = [[], [], [], [], []]
        