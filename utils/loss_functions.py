import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
# but there is a bug in the original code: it sums up the entropy over a batch. so I take mean instead of sum
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.sum(dim=1).mean(dim=0)

        return b


@torch.jit.script
def calc_energy(x: torch.Tensor, temp_factor: float = 1.0) -> torch.Tensor:
    return temp_factor * torch.logsumexp(x / temp_factor, dim=1)


class EnergyLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(EnergyLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        e = calc_energy(x, self.temp_factor)
        # energy = 1.0 / torch.linalg.vector_norm(6.0 - energy, 2)
        e = 1.0 / e.mean()
        return e


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchnorm', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def entropy(softmax: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(softmax * torch.log(softmax + 1e-6)).sum(1)


def softmax_entropy(x: torch.Tensor, temperature=1.0) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    softmax = x.softmax(1) / temperature
    return entropy(softmax)


class softmax_cross_entropy(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.NLLLoss(reduction=reduction)

    def forward(self, softmax_pred, target):
        # if target.dim() == 1:  # hard labels
        #     num_class = softmax_pred.shape[1]
        #     target = F.one_hot(target, num_classes=num_class)

        # loss = self.loss_fn(torch.log(softmax_pred + 1e-7), target)
        loss = self.loss_fn(nn.LogSoftmax(dim=1)(softmax_pred), target)
        # loss = - target * torch.log(softmax_pred + 1e-7)
        # if reduction == 'none':
        #     pass
        # elif reduction == 'mean':
        #     loss = torch.sum(loss) / softmax_pred.size(0)
        # elif reduction == 'sum':
        #     loss = torch.sum(loss, dim=1)
        return loss

@torch.jit.script
def softmax_entropy_rotta(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def JSD(p: torch.tensor, q: torch.tensor):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return torch.sum(0.5 * (F.kl_div(m, p.log(), reduction='none', log_target=True) +
                            F.kl_div(m, q.log(), reduction='none', log_target=True)), dim=1)

# def complement_CrossEntropyLoss(x, target, conf):
#     softmax = x.softmax(1)
#     complement = 1 - softmax
#     if len(target.shape) == 1:
#         target_complement= complement[torch.arange(0, len(target)), target]
#     else:
#         target_complement= complement[torch.arange(0, len(target)).unsqueeze(1), target]

#     loss = (-1 * conf * (target_complement + 1e-6).log()).mean()
#     return loss 

@torch.jit.script
def complement_CrossEntropyLoss(x, target):
    softmax = x.softmax(1)
    complement = 1 - softmax
    if len(target.shape) == 1:
        target_complement= complement[torch.arange(0, len(target)), target]
    else:
        target_complement= complement[torch.arange(0, len(target)).unsqueeze(1), target]

    loss = (-1  * (target_complement + 1e-6).log()).mean()
    return loss 

# @torch.jit.script
def weighted_CrossEntropyLoss(x, target, weight):
    softmax = x.softmax(1)
    return (-softmax[list(range(target.shape[0])), target.tolist()].log() * weight).sum()/weight.sum()


# for Energy Based Model (EBM)
def bound_max_loss(energy, bound):
    """
    return the loss value of max(0, \mathcal{F}(x) - \Delta )
    """
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()

class FreeEnergyAlignmentLoss(nn.Module):
    """
    free energy alignment loss
    """

    def __init__(self, energy_beta, energy_align_type):
        super(FreeEnergyAlignmentLoss, self).__init__()
        assert energy_beta > 0, "beta for energy calculate must be larger than 0"
        self.beta = energy_beta

        self.type = energy_align_type

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, bound):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        free_energies = -1.0 * log_sum_exp / self.beta

        bound = torch.ones_like(free_energies) * bound
        loss = self.loss(free_energies, bound)

        return loss


class NLLLoss(nn.Module):
    """
    NLL loss for energy based model
    """

    def __init__(self, energy_beta):
        super(NLLLoss, self).__init__()
        assert energy_beta > 0, "beta for energy calculate must be larger than 0"
        self.beta = energy_beta

    def forward(self, inputs, targets, soft=False):
        if soft:
            energy_c = inputs.mul(targets).sum(axis = 1).unsqueeze(-1)
        else:
            indices = torch.unsqueeze(targets, dim=1)
            energy_c = torch.gather(inputs, dim=1, index=indices)

        all_energy = -1.0 * self.beta * inputs
        free_energy = -1.0 * torch.logsumexp(all_energy, dim=1, keepdim=True) / self.beta

        nLL = energy_c - free_energy

        return nLL.mean()


class DIVLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(DIVLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        x = x.mean(axis = 0).reshape(1,-1)
        softmax = x.softmax(1)
        return (softmax * torch.log(softmax + 1e-6)).mean()
    

# MHPL # focal neighbor loss
class CrossEntropyFeatureAugWeight(nn.Module):

    """focal neighbor loss 
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyFeatureAugWeight, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weight):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if self.use_gpu: targets = targets.cuda()
        loss = (- targets * log_probs.double()).sum(dim=1)
        loss  = loss * weight
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

# MHPL entropy implementation
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

@torch.jit.script
def softmax_softlabel(x, label):
    softmax_x = x.softmax(1)
    softmax_label = label.softmax(1)
    return -(softmax_label * torch.log(softmax_x + 1e-6)).sum(1)


@torch.jit.script
def focal_CrossEntropyLoss(x, target, gamma):
    softmax = x.softmax(1)
    target_softmax = softmax[torch.arange(0, len(target)), target]
    target_focal = (1-target_softmax)**gamma
    loss = (-1 * target_focal * (target_softmax + 1e-6).log()).mean()
    return loss 

@torch.jit.script
def focal_complement_CrossEntropyLoss(x, target, conf, gamma):
    softmax = x.softmax(1)
    complement = 1 - softmax
    if len(target.shape) == 1:
        target_complement = complement[torch.arange(0, len(target)), target]
    else:
        target_complement = complement[torch.arange(0, len(target)).unsqueeze(1), target]

    target_focal = (1-target_complement)**gamma
    loss = (-1 * target_focal * (target_complement + 1e-6).log()).mean()
   
    return loss 

@torch.jit.script
def nl_softlabel(x, label):
    return -(label * torch.log(x + 1e-6)).mean()
