import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

import numpy as np
import sys

class myVGGNet(nn.Module):
    def __init__(self):
        super(myVGGNet, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature = nn.Sequential(*(vgg.features[i] for i in range(36)))
        self.feature.add_module('36: GlobalPooling', nn.AdaptiveMaxPool2d(1))
        self.classifier = nn.Sequential(
                        nn.Linear(512, 32),
                        nn.LeakyReLU(0.2),
                        nn.Linear(32, 1),
        )

        ## use multi-GPUs
        self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2])
        self.classifier = nn.DataParallel(self.classifier, device_ids=[0, 1, 2])

    def forward_feature(self, *x):
        x0 = x[0]
        x0 = self.feature(x0)
        x0 = x0.view(x0.size(0), -1)
        return x0

    def forward_train(self, *x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        x0 = self.feature(x0)
        x1 = self.feature(x1)
        x2 = self.feature(x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        o1 = self.classifier(x1 - x0)
        o2 = self.classifier(x2 - x0)
        return o2 - o1

    def forward_test(self, *x):
        feat_x = x[0]
        feat_y = x[1]

        o = self.classifier(feat_x - feat_y)

        return o

    def forward(self, *x):
        if len(x)==1:
            x = self.forward_feature(*x)
            return x
        elif len(x)==2:
            o = self.forward_test(*x)
            return o
        elif len(x)==3:
            o = self.forward_train(*x)
            return o

class myAlexNet(nn.Module):
    def __init__(self):
        super(myAlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.feature = nn.Sequential(*(alexnet.features[i] for i in range(12)))
        self.feature.add_module('12: Global Pooling', nn.AdaptiveMaxPool2d(1))
        self.classifier = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
        )

        ## use multi-GPUs
        # self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3])
        # self.classifier = nn.DataParallel(self.classifier, device_ids=[0, 1, 2, 3])

    def forward_feature(self, *x):
        x0 = x[0]
        x0 = self.feature(x0)
        x0 = x0.view(x0.size(0), -1)
        return x0

    def forward_train(self, *x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        x0 = self.feature(x0)
        x1 = self.feature(x1)
        x2 = self.feature(x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        o1 = self.classifier(x1 - x0)
        o2 = self.classifier(x2 - x0)
        return o2 - o1

    def forward_test(self, *x):
        feat_x = x[0]
        feat_y = x[1]

        o = self.classifier(feat_x - feat_y)

        return o

    def forward(self, *x):
        if len(x)==1:
            x = self.forward_feature(*x)
            return x
        elif len(x)==2:
            o = self.forward_test(*x)
            return o
        elif len(x)==3:
            o = self.forward_train(*x)
            return o

class myResNet(nn.Module):
    def __init__(self):
        super(myResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(resnet.children())[:-2])
        self.feature.add_module('8: Global Pooling', nn.AdaptiveMaxPool2d(1))
        self.classifier = nn.Sequential(
                        nn.Linear(2048, 64),
                        nn.LeakyReLU(0.2),
                        nn.Linear(64, 1),
        )

        ## use multi-GPUs
        self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3])
        self.classifier = nn.DataParallel(self.classifier, device_ids=[0, 1, 2, 3])

    def forward_feature(self, *x):
        x0 = x[0]
        x0 = self.feature(x0)
        x0 = x0.view(x0.size(0), -1)
        return x0

    def forward_train(self, *x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        x0 = self.feature(x0)
        x1 = self.feature(x1)
        x2 = self.feature(x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        o1 = self.classifier(x1 - x0)
        o2 = self.classifier(x2 - x0)
        return o2 - o1

    def forward_test(self, *x):
        feat_x = x[0]
        feat_y = x[1]

        o = self.classifier(feat_x - feat_y)

        return o

    def forward(self, *x):
        if len(x)==1:
            x = self.forward_feature(*x)
            return x
        elif len(x)==2:
            o = self.forward_test(*x)
            return o
        elif len(x)==3:
            o = self.forward_train(*x)
            return o

class tripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    ## margin = 0.1 for AlexNet

    def __init__(self, margin=0.1):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class contrastiveLoss(nn.Module):
    """
    contrastive loss
    Take embeddings of two samples and
    a target label==1 if samples are from the same class and
    label==0 otherwise
    """

    ## margin = 15.0 for AlexNet-RandomTriplet

    def __init__(self, margin=15.0):
        super(contrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output1 - output2).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances + \
                (1 - target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

if __name__ == '__main__':
    vgg = myVGG()
    alexnet = myAlexNet()
	resnet = myResNet()
    print vgg
    print alexnet
	print resnet
