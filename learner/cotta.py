import copy

import conf
from utils.memory import FIFO
from .dnn import DNN
from torch.utils.data import DataLoader
from utils import memory, loss_functions, reset_utils

from utils.loss_functions import *
import PIL
from utils import cotta_utils
import torchvision.transforms as transforms

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
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
            translate=(1/16, 1/16),
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


class CoTTA(DNN):
    def __init__(self, *args, **kwargs):
        self.mt = conf.args.ema_factor #0.999 for every dataset
        # self.rst = conf.args.restoration_factor #0.01 for all dataset
        self.rst = 0.01
        self.ap = conf.args.aug_threshold
        self.episodic = False
            
        self.transform = get_tta_transforms()

        self.src_net = None
        self.src_net_state = None
        self.net_not_ema = None

        super(CoTTA, self).__init__(*args, **kwargs)

    def init_learner(self):
        self.net.train()

        for param in self.net.parameters():  #turn on requires_grad for all
            param.requires_grad = True

        for module in self.net.modules():
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

        self.src_net_state = copy.deepcopy(self.net.state_dict())
        if conf.args.model == "resnet50_domainnet":
            self.src_net = self.deepcopy_resnet50_domainnet(self.net)
            self.net_not_ema = self.net
            self.net = self.deepcopy_resnet50_domainnet(self.net_not_ema)  # student model
        else:
            self.src_net = copy.deepcopy(self.net)
            self.net_not_ema = self.net
            self.net = copy.deepcopy(self.net_not_ema)  # student model
        for param in self.net.parameters():
            param.detach_()

        optimizer = torch.optim.SGD(
            self.net_not_ema.parameters(),
            conf.args.opt['learning_rate'],
            momentum=conf.args.opt['momentum'],
            weight_decay=conf.args.opt['weight_decay'],
            nesterov=True)
        return optimizer
        
    def model_inference(self, feats, net=None, temp=1.0, get_embedding=False):
        if net is None:
            net = self.net
        
        # self.net.train()
        # self.src_net.train()

        len_feats = len(feats)
        
        if len(feats) == 1:
            if not conf.args.use_learned_stats:
                feats = torch.concat([feats, feats])
                len_feats = 1
            else:
                net.eval()
                
        x = feats
        anchor_prob = torch.nn.functional.softmax(self.src_net(x), dim=1).max(1)[0]
        standard_ema = net(x)
        standard_ema = standard_ema[:len_feats]

        N = 32
        outputs_emas = []

        # Threshold choice discussed in supplementary
        # enable data augmentation for vision datasets
        if anchor_prob.mean(0) < self.ap:
            for i in range(N):
                outputs_ = net(self.transform(x)).detach()
                outputs_ = outputs_[:len_feats]
                outputs_emas.append(outputs_)
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        
        y_logit = outputs_ema
        y_entropy = loss_functions.softmax_entropy(y_logit)
        y_pred_softmax = F.softmax(y_logit, dim=1)
        y_conf = y_pred_softmax.max(1, keepdim=False)[0]
        y_energy = calc_energy(y_logit).cpu()
        y_pred = y_logit.max(1, keepdim=False)[1]
        
        return y_pred, y_conf, y_entropy, y_energy, None, y_pred_softmax, y_logit

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
                    self.net_not_ema.eval()
                    self.src_net.eval()
                else:
                    self.net.train()
                    self.net_not_ema.train()
                    self.src_net.train()

                feats = feats.to(device)
                outputs = self.net_not_ema(feats)

                with torch.no_grad():
                    anchor_prob = torch.nn.functional.softmax(self.src_net(feats), dim=1).max(1)[0]
                    standard_ema = self.net(feats)

                N = 32
                outputs_emas = []

                # Threshold choice discussed in supplementary
                # enable data augmentation for vision datasets
                if anchor_prob.mean(0) < self.ap:
                    for i in range(N):
                        with torch.no_grad():
                            outputs_ = self.net(self.transform(feats)).detach()
                        outputs_emas.append(outputs_)
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema

                # Student update
                self.optimizer.zero_grad()
                loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
                
                if conf.args.enable_bitta:
                    loss += self.get_bitta_ssl_loss()
                    
                loss.backward()
                self.optimizer.step()

                # Teacher update
                self.net = update_ema_variables(ema_model=self.net, model=self.net_not_ema, alpha_teacher=self.mt)

                # Stochastic restore
                if conf.args.turn_off_reset is False:
                    for nm, m in self.net_not_ema.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                rand = torch.rand(p.shape)
                                mask = (rand < self.rst).float().cuda()
                                with torch.no_grad():
                                    p.data = self.src_net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

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