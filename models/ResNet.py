'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.dropout = 0.0

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.dropout = 0.0

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, filter=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.filter = filter

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embedding=False, dropout=0.0, reverse_grad=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer2(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer3(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer4(out)
        out = F.dropout(out, p=dropout, training=True)
        out = F.avg_pool2d(out, 4)
        embeddings = out.view(out.size(0), -1)
        if reverse_grad:
            embeddings = ReverseLayerF.apply(embeddings)
        out = self.fc(embeddings)
        
        if self.filter is not None: # for imagenetR
            out = out[:, self.filter]
            
        if get_embedding:
            return out, embeddings
        return out
    
        
                    
def ResNet18(filter=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], filter=filter)


def ResNet34(filter=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], filter=filter)


def ResNet50(filter=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], filter=filter)


def ResNet101(filter=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], filter=filter)


def ResNet152(filter=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], filter=filter)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def ResNetDropout18(filter=None):
    return ResNetDropout(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], filter=filter)

def ResNetDropout50(filter=None):
    return ResNetDropout(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], filter=filter)


class ResNetDropout(torchvision.models.resnet.ResNet):
    """
    For pretrained ResNet models from Torchvision.
    """
    def __init__(self, block, num_blocks, filter=None):
        super(ResNetDropout, self).__init__(block, num_blocks)
        self.filter=filter
        
    def forward(self, x: torch.Tensor, dropout=0.0, get_embedding=False, reverse_grad=False) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer2(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer3(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer4(x)
        x = F.dropout(x, p=dropout, training=True)

        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        
        if reverse_grad:
            embedding = ReverseLayerF.apply(embedding)
            
        x = self.fc(embedding)
        
        if self.filter is not None: # for imagenetR
            x = x[:, self.filter]
        
        if get_embedding:
            return x, embedding
        else:
            return x


class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

class ResNet50_DOMAINNET(nn.Module):
    def __init__(self, bottleneck_dim=256, num_classes=126):
        super().__init__()

        # 1) ResNet backbone (up to penultimate layer)
        # if not self.use_bottleneck:
        #     pretrained = torchvision.models.resnet50(pretrained=True)
        #     model = ResNetDropout50()
        #     model.load_state_dict(pretrained.state_dict())
        #     modules = list(model.children())[:-1]
        #     self.encoder = nn.Sequential(*modules)
        #     self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        # else:
        pretrained = torchvision.models.resnet50(pretrained=True)
        model = ResNetDropout50()
        model.load_state_dict(pretrained.state_dict())
        model.fc = nn.Linear(model.fc.in_features, bottleneck_dim)
        bn = nn.BatchNorm1d(bottleneck_dim)
        self.encoder = model
        self.bn_encoder = bn
        self._output_dim = bottleneck_dim

        self.fc = nn.utils.weight_norm(nn.Linear(self._output_dim, num_classes), dim=0)

        # if checkpoint_path:
        #     self.load_from_checkpoint(checkpoint_path)
     
    def forward(self, x, dropout=0.0, get_embedding=False, reverse_grad=False):
        # 1) encoder feature
        feat = self.encoder(x, dropout=dropout, get_embedding=False, reverse_grad=reverse_grad)
        if len(x) > 1:
            feat = self.bn_encoder(feat)
        
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)
        
        if get_embedding:
            return logits, feat
        return logits

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []

        resnet = self.encoder
        for module in list(resnet.children())[:-1]:
            backbone_params.extend(module.parameters())
        # bottleneck fc + (bn) + classifier fc
        extra_params.extend(resnet.fc.parameters())
        extra_params.extend(self.bn_encoder.parameters())
        extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params
    

        
        