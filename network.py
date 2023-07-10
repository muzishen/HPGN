"""
@author:  muzishen
@contact: shenfei140721@126.com
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.resnet import resnet50, Bottleneck
from opt import opt
from torch.autograd import Variable
from torch.nn import init
import random
import torchvision.models as models
from torch.nn import Parameter
#from utils. import *
import math
import numpy as np


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
       # init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)

def genA(n):
    A = np.zeros((n ** 2, n ** 2), dtype=float)
    if n == 1:
        A[0,0] = 1
    else:
        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    A[0, 1] = 1
                    A[0, n] = 1

                elif i == 0 and j == n - 1:
                    A[n - 1, n - 2] = 1
                    A[n - 1, 2 * n - 1] = 1

                elif i == n - 1 and j == 0:
                    A[i * n, (i - 1) * n] = 1
                    A[i * n, i * n + 1] = 1

                elif i == n - 1 and j == n - 1:
                    A[n ** 2 - 1, n ** 2 - 2] = 1
                    A[n ** 2 - 1, i * n - 1] = 1

                elif i == 0:
                    A[j, i * n + j - 1] = 1
                    A[j, i * n + j + 1] = 1
                    A[j, (i + 1) * n + j ] = 1

                elif i == n - 1:
                    A[i * n + j, i * n + j - 1] = 1
                    A[i * n + j, i * n + j + 1] = 1
                    A[i * n + j, (i - 1) * n + j ] = 1

                elif j == 0:
                    A[i * n + j, (i - 1) * n + j] = 1
                    A[i * n + j, (i + 1) * n + j] = 1
                    A[i * n + j, i * n + j + 1] = 1
                elif j == n - 1:
                    A[i * n + j, (i - 1) * n + j] = 1
                    A[i * n + j, (i + 1) * n + j] = 1
                    A[i * n + j, i * n + j - 1] = 1
                else:
                    A[i * n + j, i * n + j + 1] = 1
                    A[i * n + j, i * n + j - 1] = 1
                    A[i * n + j, (i - 1) * n + j] = 1
                    A[i * n + j, (i + 1) * n + j] = 1
    return np.mat(A)

def norm(A):
    I = np.matrix(np.eye(A.shape[0]))
    D_sum = np.array(np.sum(A, axis=0))[0] 
    D = np.matrix(np.diag(D_sum)) 
    DA = (D ** -1) * A + I
    return DA



class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=256):
        super(ClassBlock, self).__init__()
        add_block1 = [nn.BatchNorm1d(input_dim), nn.ReLU(inplace=True) ,
                      nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)

        add_block2 = nn.BatchNorm1d(num_bottleneck)
        add_block2.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, class_num, bias=False)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block1(x)
        x = self.add_block2(x)
        x2 = self.classifier(x)
        return x, x2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)


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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class Spatial_GCN(nn.Module):
    def __init__(self,  in_features, out_features, dropout = False, bias=False,normalize=False):
        super(Spatial_GCN, self).__init__()
        self.adj = Parameter(torch.from_numpy(norm(genA(out_features))).float())
        self.normalize = normalize
        self.dropout = dropout
        self.bn = nn.BatchNorm2d(in_features, eps=1e-04)
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.LeakyReLU(0.2)
        self.weight = Parameter(torch.Tensor( in_features, out_features**2))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        B = x.size(0)
        X = x.view(B, self.in_features, -1)  
        AA = self.adj.expand(B,  self.out_features**2, self.out_features**2)
        support = torch.bmm(X, AA ) 
        spatial_gcn = torch.mul(support, self.weight) 
        spatial_ori = spatial_gcn.view(B, self.in_features, *x.size()[2:])
        out =  self.relu(self.bn((spatial_ori)))
        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





class HPGN(nn.Module):
    def __init__(self, num_classes):
        super(HPGN, self).__init__()
        feats = 256


        self.channel_gcn = channel_gcn(2048,1024)

        self.spatial4_gcn1  = Spatial_GCN(2048,2)
        self.spatial4_gcn2  = Spatial_GCN(2048,2)
        self.spatial4_gcn3  = Spatial_GCN(2048,2)

        self.spatial3_gcn1  = Spatial_GCN(2048,4)
        self.spatial3_gcn2  = Spatial_GCN(2048,4)
        self.spatial3_gcn3  = Spatial_GCN(2048,4)

        self.spatial2_gcn1  = Spatial_GCN(2048,8)
        self.spatial2_gcn2  = Spatial_GCN(2048,8)
        self.spatial2_gcn3  = Spatial_GCN(2048,8)


        self.spatial1_gcn1  = Spatial_GCN(2048,16)
        self.spatial1_gcn2  = Spatial_GCN(2048,16)
        self.spatial1_gcn3  = Spatial_GCN(2048,16)

        self.ap1 = nn.AvgPool2d(2,2)
        self.ap2 = nn.AvgPool2d(4, 4)
        self.ap3 = nn.AvgPool2d(8, 8)

        self.mp = nn.MaxPool2d(16, 16)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))


        self.model = ResNet()
        self.model.load_param('/home/shenfei/.cache/torch/checkpoints/resnet50-19c8e357.pth')

        self.relu = nn.LeakyReLU(0.2)
        self.reduction_1 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_4 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(feats), nn.ReLU())
        #self.reduction_4 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self.reduction_1.apply(weights_init_kaiming)
        self.reduction_2.apply(weights_init_kaiming)
        self.reduction_3.apply(weights_init_kaiming)
        self.reduction_4.apply(weights_init_kaiming)



        self.fc_id_256_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2 = nn.Linear(feats, num_classes)
        self.fc_id_256_3 = nn.Linear(feats, num_classes)
        self.fc_id_256_4 = nn.Linear(feats, num_classes)



        self.fc_id_256_1.apply(weights_init_classifier)
        self.fc_id_256_2.apply(weights_init_classifier)
        self.fc_id_256_3.apply(weights_init_classifier)
        self.fc_id_256_4.apply(weights_init_classifier)



    def forward(self, x):
        feature1 = self.model(x)
        feature2 = self.ap1(feature1)
        feature3 = self.ap2(feature1)
        feature4 = self.ap3(feature1)

        global_feature = self.gmp(feature1)
        global_tri_feature = self.reduction_1(global_feature).squeeze(dim=3).squeeze(dim=2)
        global_f_feature = self.fc_id_256_1(global_tri_feature)

        gcn1_learning1 = self.spatial1_gcn1(feature1)
        gcn1_learning2 = self.spatial1_gcn2(gcn1_learning1)
        gcn1_learning3 = self.spatial1_gcn3(gcn1_learning2)
        gcn1_gap = self.gap(gcn1_learning3)
        #
        #
        gcn2_learning1 = self.spatial2_gcn1(feature2)
        gcn2_learning2 = self.spatial2_gcn2(gcn2_learning1)
        gcn2_learning3 = self.spatial2_gcn3(gcn2_learning2)
        gcn2_gap = self.gap(gcn2_learning3)
        #
        gcn3_learning1 = self.spatial3_gcn1(feature3)
        gcn3_learning2 = self.spatial3_gcn2(gcn3_learning1)
        gcn3_learning3 = self.spatial3_gcn3(gcn3_learning2)
        gcn3_gap = self.gap(gcn3_learning3)

        gcn4_learning1 = self.spatial4_gcn1(feature4)
        gcn4_learning2 = self.spatial4_gcn2(gcn4_learning1)
        gcn4_learning3 = self.spatial4_gcn3(gcn4_learning2)
        gcn4_gap = self.gap(gcn4_learning3)

        # 2 + 4 + 8 + 16
        gcn_gap =  gcn4_gap + gcn3_gap + gcn2_gap + gcn1_gap

        gcn_tri_feature = self.reduction_2(gcn_gap).squeeze(dim=3).squeeze(dim=2)

        gcn_f_feature = self.fc_id_256_2(gcn_tri_feature)

        predict = torch.cat([global_tri_feature, gcn_tri_feature,], dim=1)

        return  predict, predict, global_f_feature, gcn_f_feature

if __name__ == '__main__':
    net = HPGN(num_classes=576)
    print(sum(param.numel() for param in net.parameters()))
    input = Variable(torch.FloatTensor(64, 3, 256, 256))
    x = HPGN.forward(net, input)

