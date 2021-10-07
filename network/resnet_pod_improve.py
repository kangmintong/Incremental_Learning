# compared with resnet_lucir.py
# remove ReLU in the last block of every layer
# all attention: output features after every block; not all attention: output features after every layer

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

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

class Stage(nn.Module):
    def __init__(self,blocks, block_relu=False):
        super(Stage, self).__init__()
        self.blocks=nn.ModuleList(blocks)
        self.block_relu=block_relu
        self.gradients=[]

    def save_gradient(self,grad):
        self.gradients.append(grad)

    def forward(self,x):
        self.gradients=[]
        intermediary_features = []
        for b in self.blocks:
            x = b(x)
            intermediary_features.append(x)
            if self.block_relu:
                x = F.relu(x)
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        return intermediary_features, x

class multi_fc(nn.Module):
    def __init__(self, fc_num, in_feature, out_feature, device):
        super(multi_fc, self).__init__()
        self.fcs=[]
        self.device=device
        self.fc_num=fc_num
        self.in_features=in_feature
        self.out_features=out_feature
        self.fc1s=[]
        for i in range(fc_num):
            self.fcs.append(CosineLinear(in_feature, out_feature).to(self.device))
        self.fcs=nn.ModuleList(self.fcs)
    def forward(self,x):
        cnt=0
        for fc in self.fcs:
            if cnt==0:
                y=fc(x)
                cnt += 1
            else:
                y = y + fc(x)
        return y
    def __getitem__(self, item):
        cnt=0
        for fc in self.fcs:
            if cnt==item:
                return fc
            else:
                cnt += 1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, fc_num=10, device=None):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.device=device
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2]-1, stride=2)
        self.last_conv = BasicBlock(self.inplanes, 64)
        self.avgpool = nn.AvgPool2d(8, stride=1)

        self.multi_fc=multi_fc(fc_num,64 * block.expansion, num_classes, device)

        self.gradients=[]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Stage(layers)

    # def forward(self, x, features=False, last_feature=False):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     features1, x = self.layer1(x)
    #     features2, x = self.layer2(x)
    #     features3, x = self.layer3(x)
    #     x = self.last_conv(x)
    #     features4 = x
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     if last_feature:
    #         return x
    #
    #     y = self.multi_fc(x)
    #
    #     if features:
    #         features = [features1[-1], features2[-1], features3[-1], features4]
    #         return features, y
    #     return y

    def save_gradient(self,grad):
        self.gradients.append(grad)

    def forward(self, x, features=False, last_feature=False):
        self.gradients=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features1, x = self.layer1(x)
        features2, x = self.layer2(x)
        features3, x = self.layer3(x)
        x = self.last_conv(x)
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        features4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if last_feature:
            return x

        y = self.multi_fc(x)

        if features:
            features = [features1[-1], features2[-1], features3[-1], features4]
            return features, y
        return y

def resnet20(pretrained=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model