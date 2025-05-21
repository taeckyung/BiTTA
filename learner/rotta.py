from typing import Optional, Tuple

import conf
from utils.memory import CSTU
from .dnn import DNN
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.loss_functions import *
from utils import bn_layers_rotta
from copy import deepcopy
from utils.custom_transforms import get_tta_transforms
from torch import nn

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class RoTTA(DNN):
    def __init__(self, *args, **kwargs):
        self.nu = 0.001
        self.alpha = 0.05
        self.net_not_ema = None  # TBU in init_learner
        self.transform = get_tta_transforms(conf.args.dataset)
        super(RoTTA, self).__init__(*args, **kwargs)
        assert isinstance(self.mem, CSTU)
        

    def init_learner(self):
        self.net.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in self.net.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
                
            if isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
                sub_module.requires_grad_(True)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.net, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = bn_layers_rotta.RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = bn_layers_rotta.RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.alpha)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.net, name, momentum_bn)

        params, param_names = self.collect_params(self.net)
        optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

        if conf.args.model == "resnet50_domainnet":
            net_ema = self.deepcopy_resnet50_domainnet(self.net)  # student model
        else:
            net_ema = deepcopy(self.net)
            
        for param in net_ema.parameters():
            param.detach_()

        self.net_not_ema = self.net
        self.net = net_ema  # set self.net to self.net_ema
        self.net.to(device)

        return optimizer
        
    # def set_target_data(self, *args, **kwargs):
    #     super(RoTTA, self).set_target_data(*args, **kwargs)
        # self.transform = get_tta_transforms(tuple(self.target_train_set[0][0].shape[1:]))
       

    def test_time_adaptation(self):
        assert isinstance(self.mem, CSTU)

        feats, ages, labels = self.mem.get_memory()

        feats = torch.stack(feats).to(device)
        ages = torch.Tensor(ages).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(feats, labels, ages)
        data_loader = DataLoader(dataset, batch_size=conf.args.tta_batch_size,
                                 shuffle=True, drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats, _, ages) in enumerate(data_loader):
                # setup models
                self.net_not_ema.train()
                self.net.train()

                if len(feats) == 1:  # avoid BN error
                    self.net_not_ema.eval()
                    self.net.eval()

                strong_sup_aug = self.transform(feats)
                ema_sup_out = self.net(feats)
                stu_sup_out = self.net_not_ema(strong_sup_aug)
                instance_weight = self.timeliness_reweighting(ages)
                loss = (softmax_entropy_rotta(stu_sup_out, ema_sup_out) * instance_weight).mean()
                
                if conf.args.enable_bitta:
                    loss += self.get_bitta_ssl_loss()
                # RoTTA backprops on student model. No gradient in teacher.
                if loss is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.net = self.update_ema_variables(self.net, self.net_not_ema, self.nu)

    def timeliness_reweighting(self, ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))

    def update_ema_variables(self, ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
    
    def deepcopy_resnet50_domainnet(self, net):
        from models import ResNet
        copied_net = ResNet.ResNet50_DOMAINNET().to(device)
        copied_net = torch.nn.Sequential(deepcopy(net[0]), copied_net)
        
        copied_net.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in copied_net.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
                
            if isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
                sub_module.requires_grad_(True)

        for name in normlayer_names:
            bn_layer = get_named_submodule(copied_net, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = bn_layers_rotta.RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = bn_layers_rotta.RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.alpha)
            momentum_bn.requires_grad_(True)
            set_named_submodule(copied_net, name, momentum_bn)
                
        copied_net.load_state_dict(deepcopy(net.state_dict()))
        return copied_net
    
def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)
            
            