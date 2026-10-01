import os
import copy
import PIL
import torch.nn
from torch import optim
from torch.nn import KLDivLoss
from torchvision import transforms
from torch.utils.data import DataLoader

import conf
from utils import cotta_utils, mecta
from .dnn import DNN
from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class BiTTA(DNN):
    """
    BiTTA (Binary Test-Time Adaptation) implementation.
    
    This class implements the BiTTA method for test-time adaptation, which uses
    binary feedback (correct/wrong) from actively selected samples to adapt
    the model during test time. It combines two types of losses:
    - BFA (Binary Feedback Adaptation) loss for actively labeled samples
    - ABA (Agreement-Based Adaptation) loss for unlabeled samples
    
    The implementation is inspired by:
    - SoTTA: https://github.com/taeckyung/SoTTA
    - FlatMatch: https://github.com/tmllab/2023_NeurIPS_FlatMatch
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize BiTTA model. Refer `init_learner()` for actual initialization.
        """
        self.atta_src_net = None
        super(BiTTA, self).__init__(*args, **kwargs)
        assert (conf.args.memory_type in ["ActivePriorityFIFO", "ActivePriorityPBRS"])

    def reset(self):
        """Reset the BiTTA model to its initial state."""
        super(BiTTA, self).reset()

    def init_learner(self):
        """
        Initialize the learner by setting up parameters for adaptation.
        
        This method configures which parameters should be adapted during test-time,
        sets up BatchNorm layer behavior, and initializes the optimizer.
        
        Returns:
            torch.optim.Optimizer: Configured SGD optimizer for adaptation
        """
        # Enable gradients for all parameters
        for param in self.net.parameters():
            param.requires_grad = True

        for name, module in self.net.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        if conf.args.enable_mecta:  # for additional study
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
        """
        Perform test-time adaptation using BiTTA algorithm.
        
        This method implements the core BiTTA adaptation procedure:
        1. Prepare data loaders for unlabeled, correct, and wrong samples
        2. Apply Monte Carlo Dropout for uncertainty estimation
        3. Compute BFA and ABA losses
        4. Update model parameters
        5. Apply stochastic restoration if enabled
        """
        # Prepare unlabeled data
        u_feats, u_labels, _, _ = self.mem.get_u_memory()
        if len(u_feats) != 0:
            u_feats = torch.stack(u_feats).to(device)
            u_labels = torch.tensor(u_labels).to(device)
            u_dataset = torch.utils.data.TensorDataset(u_feats, u_labels)
            u_dataloader = DataLoader(u_dataset, batch_size=conf.args.update_every_x,
                                      shuffle=True, drop_last=False, pin_memory=False)
        else:
            u_dataloader = [(None, None)]

        # Prepare correct samples
        correct_feats, correct_labels, _, _ = self.mem.get_correct_memory()
        self.json_active['num_correct_mem'] += [len(correct_labels)]
        if len(correct_feats) != 0:
            correct_feats = torch.stack(correct_feats).to(device)
            correct_labels = torch.tensor(correct_labels).to(device)

            correct_dataset = torch.utils.data.TensorDataset(correct_feats, correct_labels)
            correct_dataloader = DataLoader(correct_dataset, batch_size=conf.args.update_every_x,
                                            shuffle=True, drop_last=False, pin_memory=False)
        else:
            correct_dataloader = [(None, None)]

        # Prepare wrong samples
        wrong_feats, wrong_labels, wrong_gt_labels, _ = self.mem.get_wrong_memory()
        self.json_active['num_wrong_mem'] += [len(wrong_labels)]
        if len(wrong_feats) != 0:
            wrong_feats = torch.stack(wrong_feats).to(device)
            wrong_labels = torch.tensor(wrong_labels).to(device)
            wrong_gt_labels = torch.tensor(wrong_gt_labels, dtype=torch.long).to(device)

            wrong_dataset = torch.utils.data.TensorDataset(wrong_feats, wrong_labels, wrong_gt_labels)
            wrong_dataloader = DataLoader(wrong_dataset, batch_size=conf.args.update_every_x,
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

                # Apply Monte Carlo Dropout for optimization
                mcd_mean_softmax = self.dropout_inference(data, conf.args.n_dropouts, dropout=conf.args.dropout_rate)

                with torch.no_grad():
                    outputs = self.net(data)
                    outputs_softmax = outputs.softmax(dim=1)

                # Compute BiTTA losses (BFA and ABA)
                bfa_loss, aba_loss = self.get_loss(mcd_mean_softmax, outputs_softmax, correct_labels_, wrong_labels_, u_labels_)

                # Final loss combination
                loss = conf.args.w_final_bfa_loss * bfa_loss + conf.args.w_final_aba_loss * aba_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # If needed, apply stochastic restoration for better stability
        # Only applied in Tiny-ImageNet-C
        if conf.args.restoration_factor > 0:
            for nm, m in self.net.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        rand = torch.rand(p.shape)
                        mask = (rand < conf.args.restoration_factor).float().cuda()
                        with torch.no_grad():
                            p.data = self.src_net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

    def get_loss(self, mc_dropout_softmax, original_softmax, correct_labels_, wrong_labels_, u_labels_):
        """
        Compute BiTTA losses for different types of samples.
        
        This method implements the loss computation described in Algorithm 1:
        - BFA (Binary Feedback Adaptation) loss for correct/wrong samples (Line 20)
        - ABA (Agreement-Based Adaptation) loss for unlabeled samples (Lines 22-25)
        
        Args:
            mc_dropout_softmax: Monte Carlo dropout softmax predictions
            original_softmax: Original model softmax predictions  
            correct_labels_: Labels for correctly predicted samples
            wrong_labels_: Labels for wrongly predicted samples
            u_labels_: Labels for unlabeled samples
            
        Returns:
            tuple: (correct_loss, wrong_loss, unlabeled_loss)
        """
        bfa_loss = torch.tensor([0.0]).to(device)
        aba_loss = torch.tensor([0.0]).to(device)

        # BFA loss for correct samples (reward: +1)
        if correct_labels_ is not None:
            correct_dropout_outputs = mc_dropout_softmax[:len(correct_labels_)]
            if conf.args.use_original_conf:
                correct_dropout_outputs = original_softmax[:len(correct_labels_)]
            bfa_loss = self.class_criterion(correct_dropout_outputs, correct_labels_)
            
        # BFA loss for wrong samples (reward: -1)
        if wrong_labels_ is not None:
            start_idx = len(correct_labels_) if correct_labels_ is not None else 0
            end_idx = -len(u_labels_) if u_labels_ is not None else len(mc_dropout_softmax)

            own_wrong_dropout_outputs = mc_dropout_softmax[start_idx:end_idx]
            bfa_loss += -self.class_criterion(own_wrong_dropout_outputs, wrong_labels_)

        # ABA loss for unlabeled samples (reward: +1 only for confident samples)
        if u_labels_ is not None:
            start_idx = len(correct_labels_) if correct_labels_ is not None else 0
            start_idx += len(wrong_labels_) if wrong_labels_ is not None else 0

            own_u_dropout_outputs = mc_dropout_softmax[start_idx:]
            total_u_dropout_outputs = mc_dropout_softmax[start_idx:]

            original_pred = original_softmax[start_idx:].argmax(dim=1).detach()
            confident_sample_idx = original_pred == total_u_dropout_outputs.argmax(dim=1)
            aba_loss = self.class_criterion(own_u_dropout_outputs[confident_sample_idx], original_pred[confident_sample_idx])

        return bfa_loss, aba_loss

    def pre_active_sample_selection(self):
        """
        Prepare for active sample selection by updating BatchNorm statistics.
        
        This implements Algorithm 1: Line 15, which updates the running statistics
        of BatchNorm layers before performing active sample selection.
        """
        self.enable_running_stats()
        
        # Update BN statistics
        self.net.train()
        feats, _, _ = self.fifo.get_memory()
        feats = torch.stack(feats).to(device)
        with torch.no_grad():
            _ = self.net(feats)

        self.disable_running_stats()
