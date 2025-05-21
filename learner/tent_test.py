import torch.nn
from torch import optim

import conf
from utils.memory import FIFO
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TENT(DNN):
    def __init__(self, *args, **kwargs):
        super(TENT, self).__init__(*args, **kwargs)

    def init_learner(self):
        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        optimizer = torch.optim.SGD(
                        self.net.parameters(),
                        conf.args.opt['learning_rate'],
                        momentum=conf.args.opt['momentum'],
                        weight_decay=conf.args.opt['weight_decay'],
                        nesterov=True)
        # optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'], weight_decay=0.0)
        return optimizer

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

                entropy_loss = HLoss()

                if not conf.args.dropout_softlabel:
                    
                    preds_of_data = self.net(feats)

                    # loss = entropy_loss(preds_of_data)
                    
                    with torch.no_grad():
                        _, mc_dropout_soft_labels, _  = self.dropout_inference(feats, 10, dropout=conf.args.dropout_rate)
                    
                    if conf.args.activate_filter:
                        filter_idx = (mc_dropout_soft_labels > 1 / conf.args.opt['num_class'])

                        new_u_output = mc_dropout_soft_labels.clone().detach()
                        new_u_output[~filter_idx] = 0.0
                        new_u_output = F.normalize(new_u_output, p=1, dim=1)
                    
                        loss = nl_softlabel(preds_of_data.softmax(1), new_u_output)
                    else:
                        loss = nl_softlabel(preds_of_data.softmax(1), mc_dropout_soft_labels)

                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                
                else:
                    # conf.args.dropout_softlabel
                    with torch.no_grad():
                        _, mc_dropout_soft_labels, _  = self.dropout_inference(feats, 10, dropout=conf.args.dropout_rate)
                
                    self.optimizer.zero_grad()
                    for n_iter in range(conf.args.n_dropouts):
                        _, mc_dropout_softmax, _ = self.dropout_inference(feats, 1, dropout=conf.args.dropout_rate)

                        mc_dropout_softmax = (mc_dropout_soft_labels * 10 + mc_dropout_softmax)/11
                        
                        if conf.args.activate_filter:
                            filter_idx = (mc_dropout_soft_labels > 1 / conf.args.opt['num_class'])

                            new_u_output = mc_dropout_soft_labels.clone().detach()
                            new_u_output[~filter_idx] = 0.0
                            new_u_output = F.normalize(new_u_output, p=1, dim=1)

                            unlabeled_loss = nl_softlabel(mc_dropout_softmax, new_u_output)
                        else:
                            unlabeled_loss = nl_softlabel(mc_dropout_softmax, mc_dropout_soft_labels)

                        loss = unlabeled_loss
                        loss *= 11/conf.args.n_dropouts
                        
                        loss.backward()
                        
                    self.optimizer.step()
