import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR

import os
import sys
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
import time
import argparse
import json
import csv
import visdom

import net
from net import tripletLoss, NpairLoss, weight_init
from dataload import triTrainDataset, triNpTrainDataset
from dataset import buildTestData

class Config():
    ''' global parameters '''

    # test_dataset = 'oxf'
    test_dataset = 'par'

    pair_num = 11 * 2000

    if test_dataset == 'oxf':
        train_txt = '/data4/fong/pytorch/RankNet/building/train_oxf.txt'
        img_testfolder = '/data4/fong/pytorch/RankNet/building/test_oxf'
        img_testpath = '/data4/fong/pytorch/RankNet/building/test_oxf/images'
        gt_path = '/data4/fong/oxford5k/oxford5k_groundTruth'
    if test_dataset == 'par':
        train_txt = '/data4/fong/pytorch/RankNet/building/train_par.txt'
        img_testfolder = '/data4/fong/pytorch/RankNet/building/test_par'
        img_testpath = '/data4/fong/pytorch/RankNet/building/test_par/images'
        gt_path = '/data4/fong/paris6k/paris_groundTruth'

    eval_func = '/data4/fong/oxford5k/evaluation/compute_ap'
    retrieval_result = '/data4/fong/pytorch/RankNet/building/retrieval'

    building = buildTestData(img_path=img_testpath, gt_path=gt_path, eval_func=eval_func)

    img_transform = transforms.Compose([
        ## warning: if the dataloader is not customized,
        ## images in one batch shoube be of the same size.

        ## Scale(l): smaller edge of image will be matched to l
        transforms.Scale((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    batch_size = 8
    train_epoch = 20
    learning_rate1 = 1e-5
    learning_rate2 = 5e-4

def train(net_type):

    print "========== Siamese-Rank =========="
    train_data = triNpTrainDataset(txt=Config.train_txt, \
        pair_num=Config.pair_num, transform=Config.img_transform)

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4, batch_size=Config.batch_size)

    test_data = ImageFolder(Config.img_testfolder, transform=Config.img_transform)
    test_dataloader = DataLoader(dataset=test_data, shuffle=False, num_workers=4, batch_size=Config.batch_size)

    if net_type == 'A':
        model = net.myAlexNet()
        print "---------- AlexNet ----------"
    if net_type == 'V':
        model = net.myVGGNet()
        print "---------- VGGNet ----------"
    if net_type == 'R':
        model = net.myResNet()
        print "---------- ResNet ----------"

    ## weights initialization
    # model.apply(weight_init)

    criterion = NpairLoss()

    optimizer = optim.Adam([
        {'params': model.feature.parameters(), 'lr': Config.learning_rate1},
        {'params': model.classifier.parameters(), 'lr': Config.learning_rate2}
    ])

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        criterion = criterion.cuda()

    iter_count = 1
    check_step = 100
    train_loss = 0.0
    check_loss = []
    check_accuracy = []

    for e in range(Config.train_epoch):

        ## test
        model.eval()
        feature_map = torch.FloatTensor()
        for data in tqdm(test_dataloader):
            img, _ = data
            if use_gpu:
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            feature = model(img)
            feature = F.normalize(feature, p=2, dim=1)
            feature_map = torch.cat((feature_map, feature.cpu().data), 0)
        feature_map = feature_map.numpy()
        similarity = np.dot(feature_map, feature_map.T)
        print "test size: ", feature_map.shape
        e_accuracy = Config.building.evalRetrieval(similarity, Config.retrieval_result)
        check_accuracy.append(e_accuracy)
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'epoch: {}, accuracy: {:.8f}'.format(e, e_accuracy)

        model.train()
        scheduler.step()
        for data in train_dataloader:
            anchor, positive, label = data
            if use_gpu:
                anchor = Variable(anchor).cuda()
                positive = Variable(positive).cuda()
                label = Variable(label).cuda()
            else:
                anchor = Variable(anchor)
                positive = Variable(positive)
                label = Variable(label)

            feat0 = model(anchor)
            feat1 = model(positive)

            feat_diff = feat1.view(1, feat1.size(0), feat1.size(1)).expand(feat1.size(0), feat1.size(0), feat1.size(1)) - \
                    feat0.view(feat0.size(0), 1, feat0.size(1)).expand(feat0.size(0), feat0.size(0), feat0.size(1))
            feat_diff = feat_diff.view(feat_diff.size(0)*feat_diff.size(1), -1)
            out_n2 = model.classifier(feat_diff)
            out_n2 = out_n2.view(feat0.size(0), feat0.size(0))
            # out_n = model.classifier(feat1 - feat0)
            # out = out_n2 - out_n.repeat(1, out_n2.size(1))
            out = out_n2

            label = label.view(label.size(0), -1)
            label = (label == torch.transpose(label, 0, 1)).float()
            label = label / torch.sum(label, dim=1, keepdim=True).float()

            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data[0]
            if iter_count % check_step == 0:
                check_loss.append(train_loss/check_step)
                print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                train_loss = 0.0

            iter_count += 1
    
    vis = visdom.Visdom(env=Config.test_dataset)
    vis.line(
            X = np.arange(1, len(check_loss)+1, 1) * check_step,
            Y = np.array(check_loss),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='itr',
                ylabel='loss'
            )
    )
    vis.line(
            X = np.arange(1, len(check_accuracy)+1, 1),
            Y = np.array(check_accuracy),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='epoch',
                ylabel='mAP'
            )
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dataset: Paris6k, Oxford5k')
    parser.add_argument('-n', '--net', type=str, required=False, help='CNN: AlexNet(\'A\', default), VGGNet(\'V\'), ResNet(\'R\')')
    parser.set_defaults(net='A')
    args = parser.parse_args()
    print 'Parameters: ', args

    train(net_type=args.net)
