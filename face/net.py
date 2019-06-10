import torch
from torch import nn
import torch.nn.init as nninit
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

import numpy as np
import sys

def weight_init(m):
    if type(m) == nn.Linear:
        nninit.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nninit.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        nninit.constant(m.bias, 0.01)


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
                        nn.LeakyReLU(0.2)
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

class myAlexNet(nn.Module):
    def __init__(self):
        super(myAlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=True) ## train from scratch
        self.feature = nn.Sequential(*(alexnet.features[i] for i in range(12)))
        self.feature.add_module('12: Global Pooling', nn.AdaptiveMaxPool2d(1))
        self.classifier = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
        )
        self.fc = nn.Sequential(
                nn.Linear(256, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10575)
        )

        ## use multi-GPUs
        # self.feature = nn.DataParallel(self.feature, device_ids=[0, 1, 2, 3])
        # self.classifier = nn.DataParallel(self.classifier, device_ids=[0, 1, 2, 3])
        # self.fc = nn.DataParallel(self.classifier, device_ids=[0, 1, 2, 3])

    def forward_feature(self, *x):
        x0 = x[0]
        x0 = self.feature(x0)
        x0 = x0.view(x0.size(0), -1)
        return x0

    def forward_classify(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

    def forward_rank_softmax(self, x0, x1, x2):
        x0 = self.feature(x0)
        x1 = self.feature(x1)
        x2 = self.feature(x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        o1 = self.classifier(x1 - x0)
        o2 = self.classifier(x2 - x0)

        z0 = self.fc(x0)
        z1 = self.fc(x1)
        z2 = self.fc(x2)

        return o2 - o1, z0, z1, z2

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
        return o1, o2, x0, x1, x2

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
            o1, o2, x0, x1, x2 = self.forward_train(*x)
            return o1, o2, x0, x1, x2

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
                        nn.LeakyReLU(0.2)
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

class faceNet(nn.Module):
    def __init__(self):
        super(faceNet, self).__init__()
        self.feature = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 96, kernel_size=3, padding=1),
                    nn.Conv2d(96, 192, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(192, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 160, kernel_size=3, padding=1),
                    nn.Conv2d(160, 320, kernel_size=3, padding=1),
                    nn.AvgPool2d(kernel_size=6, stride=1),
        )
        self.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(320, 10575),
        )
        self.classifier = nn.Sequential(
                        nn.Linear(320, 128),
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

    def forward_classify(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

    def forward_rank_softmax(self, x0, x1, x2):
        x0 = self.feature(x0)
        x1 = self.feature(x1)
        x2 = self.feature(x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        o1 = self.classifier(x1 - x0)
        o2 = self.classifier(x2 - x0)

        z0 = self.fc(x0)
        z1 = self.fc(x1)
        z2 = self.fc(x2)

        return o2 - o1, z0, z1, z2

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
        return o1, o2, x0, x1, x2

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
            o1, o2, x0, x1, x2 = self.forward_train(*x)
            return o1, o2, x0, x1, x2

class tripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
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
        # print output1
        # print distances.mean()
        # sys.exit(0)
        # losses = 0.5 * (target.float() * distances + \
        #         (1 - target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        losses = 0.5 * (target.float() * distances + (1 - target).float() * F.relu(self.margin - distances))
        return losses.mean() if size_average else losses.sum()


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, 1), 1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, 1), 1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=3e-3):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss
        return loss

        
if __name__ == '__main__':
    vgg = myVGG()
    alexnet = myAlexNet()
    print vgg
    print alexnet
