"""
WRN architecture (https://arxiv.org/abs/1605.07146)
Code adapted from (https://github.com/JerryYLi/bg-resample-ood/blob/master/models/wide_resnet.py)
"""

from xml.etree.ElementInclude import include
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_size=32, clf_type='inner', include_binary=True):
        super(Wide_ResNet, self).__init__()
        self.clf_type = clf_type # ['inner', 'euclidean', 'cosine']
        self.include_binary = include_binary
        self.in_planes = 16

        assert ((depth-4)%6 == 0), 'Wide ResNet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        # print('Wide ResNet %dx%d' % (depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        if self.clf_type == 'inner':
            self.linear = nn.Linear(nStages[3], num_classes)
            nn.init.kaiming_normal_(self.linear.weight.data, nonlinearity='relu')
            self.linear.bias.data = torch.zeros(size=self.linear.bias.size())
        elif self.clf_type == 'euclidean':
            self.linear = nn.Linear(nStages[3], num_classes, bias=False)
            nn.init.kaiming_normal_(self.linear.weight.data, nonlinearity='relu')
        else:
            raise RuntimeError('<<< Invalid CLF TYPE: {}'.format(self.clf_type))
        # binary classification head
        if self.include_binary:
            self.binary_linear = nn.Linear(1, 1) # energy as variable
        self.feature_dim = self.linear.in_features

        self.pool_size = in_size // 4

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, ret_feat=False, ret_el=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.pool_size)
        out = out.view(out.size(0), -1)
        if self.clf_type == 'inner':
            logit = self.linear(out)
        elif self.clf_type == 'euclidean':
            out_ = out.unsqueeze(2) # [BATCH, FEAT_DIM, 1]
            w = self.linear.weight.T.unsqueeze(0) # [1, FEAT_DIM, NUM_CLA]
            # print(torch.norm(out, dim=1))
            # print(self.linear.weight.shape) # [NUM_CLA, FEAT_DIM]
            # print(torch.norm(self.linear.weight, dim=1))
            # print(torch.max(torch.matmul(norm(out), norm(self.linear.weight).T), dim=1)[0]) # [BATCH, NUM_CLA]
            # exit()
            logit = -((out_ - w).pow(2)).mean(1)
        else:
            raise RuntimeError('<<< Invalid CLF TYPE: {}'.format(self.clf_type))
        
        if self.include_binary:
            # energy logit
            energy = -torch.logsumexp(logit, dim=1).unsqueeze(1) # [BATCH, 1]
            # print(energy.size())
            # exit()
            energy_logit = self.binary_linear(energy) # [BATCH, 1]
            # w = torch.abs(self.binary_linear.weight.T) # [FEAT_DIM, 1]
            # energy_logit = torch.mm(energy, w) + self.binary_linear.bias # [BATCH, 1]
            # print(self.binary_linear.weight, self.binary_linear.bias)

            if ret_feat:
                if ret_el:
                    return logit, out, energy_logit
                else:
                    return logit, out
            return logit
        
        if ret_feat:
            return logit, out
        else:
            return logit

if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())