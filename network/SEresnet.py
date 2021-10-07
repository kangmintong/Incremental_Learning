import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class cifar_SE_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(cifar_SE_block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if planes!=inplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class Stage(nn.Module):
    def __init__(self,blocks, block_relu=False):
        super(Stage, self).__init__()
        self.blocks=nn.ModuleList(blocks)
        self.block_relu=block_relu
    def forward(self,x):
        intermediary_features = []
        for b in self.blocks:
            x = b(x)
            intermediary_features.append(x)
            if self.block_relu:
                x = F.relu(x)
        return intermediary_features, x

class cifar_SE_resnet(nn.Module):
    def __init__(self, block, layers, num_classes=100, reduction=16):
        super(cifar_SE_resnet,self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=layers[2], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = CosineLinear(64, num_classes)
        self.initialize()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, reduction))
            self.inplanes = planes
        return Stage(layers)

    def forward(self, x, features=False, last_feature=False, all_attention=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features1, x = self.layer1(x)
        features2, x = self.layer2(x)
        features3, x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = x
        if last_feature:
            return feature
        x = self.fc(x)
        if features:
            if all_attention:
                features = [*features1,*features2,*features3]
                return features, x
            else:
                features = [features1[-1], features2[-1], features3[-1]]
                return features, x
        return x

def resnet32(**kwargs):
    model = cifar_SE_resnet(cifar_SE_block, [5, 5, 5], **kwargs)
    return model
