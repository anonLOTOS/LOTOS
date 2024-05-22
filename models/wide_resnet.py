import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, bn=True, device='cuda', elu_flag=False):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn = bn   
        self.elu_flag = elu_flag

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        if self.bn:
            if self.elu_flag:
                out = self.dropout(self.conv1(F.elu(self.bn1(x))))
            else:
                out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        else:
            if self.elu_flag:
                out = self.dropout(self.conv1(F.elu(x)))
            else:
                out = self.dropout(self.conv1(F.relu(x)))

        if self.bn:
            if self.elu_flag:
                out = self.conv2(F.elu(self.bn2(out)))
            else:
                out = self.conv2(F.relu(self.bn2(out)))
        else:
            if self.elu_flag:
                out = self.conv2(F.elu(out))
            else:
                out = self.conv2(F.relu(out))

        out += self.shortcut(x)

        return out

class Wide_ResNet_orig(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_chan=3, bn=True, device='cuda', elu_flag=False):
        super(Wide_ResNet_orig, self).__init__()
        self.in_planes = 16

        self.bn = bn
        self.elu_flag = elu_flag

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(in_chan, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, bn=bn, device=device, elu_flag=elu_flag)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, bn=bn, device=device, elu_flag=elu_flag)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, bn=bn, device=device, elu_flag=elu_flag)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bn=True, device='cuda', elu_flag=False):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, bn=bn, device=device, elu_flag=elu_flag))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.bn:
            if self.elu_flag:
                out = F.elu(self.bn1(out))
            else:
                out = F.relu(self.bn1(out))
        else:
            if self.elu_flag:
                out = F.elu(out)
            else:
                out = F.relu(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def wideResnet_orig(in_chan=3, depth=34, widen_factor=1, dropout_rate=0.0, num_classes=10, bn=True, bn_clip=False, bn_count=100, bn_hard=False, clip_linear=False, device='cuda', elu_flag=False):
    return Wide_ResNet_orig(depth, widen_factor, dropout_rate, num_classes, in_chan=in_chan, bn=bn, device=device, elu_flag=elu_flag)

