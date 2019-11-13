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
import visdom
import argparse

import net
from net import tripletLoss, contrastiveLoss
from dataload import myTriDataset_random, myTriDataset_anchor, myContrasDataset

class Config():
    ''' global parameters '''

    anchorDir = '/path/to/anchors'
    pair_num = 6200 * 20

    img_trainpath = '/path/to/train'
    train_txt = '/path/to/train.txt'
    img_testpath = '/path/to/test'

    img_transform = transforms.Compose([
        transforms.Resize(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ## triplet network: batch_size*3 images in each batch
    batch_size = 64
    train_epoch = 30
    learning_rate1 = 5e-5
    learning_rate2 = 1e-4

def computeScore(feature_map):
    ''' compute NS-score '''

    similarity = np.dot(feature_map, feature_map.T)
    rank = np.argsort(-similarity, axis=1)
    score = 0.0
    for row in range(feature_map.shape[0]):
        for col in range(4):
            if rank[row, col]/4 == row/4:
                score += 1.0
    NS_score = score/float(feature_map.shape[0])
    return NS_score

def train(siamese_mode, net_type, margin):

    if siamese_mode == 'rank':
        print "========== siamese-rank =========="
        train_data = myTriDataset_anchor(img_rootpath=Config.img_trainpath, txt=Config.train_txt, \
            anchorDir=Config.anchorDir, pair_num=Config.pair_num, transform=Config.img_transform)
        # train_data = myTriDataset_random(img_rootpath=Config.img_trainpath, txt=Config.train_txt, \
            # transform=Config.img_transform)

    elif siamese_mode == 'triplet':
        print "========== siamese-triplet =========="
        train_data = myTriDataset_anchor(img_rootpath=Config.img_trainpath, txt=Config.train_txt, \
            anchorDir=Config.anchorDir, pair_num=Config.pair_num, transform=Config.img_transform)
        # train_data = myTriDataset_random(img_rootpath=Config.img_trainpath, txt=Config.train_txt, \
            # transform=Config.img_transform)

    elif siamese_mode == 'contrastive':
        print "========== siamese-contrastive =========="
        train_data = myContrasDataset(img_rootpath=Config.img_trainpath, txt=Config.train_txt, \
            transform=Config.img_transform)
    else:
        print "You should set siamese mode as \'triplet\' or \'constracitve\'."
        # sys.exit(0)
        return

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4, batch_size=Config.batch_size)

    test_data = ImageFolder(Config.img_testpath, transform=Config.img_transform)
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

    if siamese_mode == 'rank':
        criterion = nn.BCEWithLogitsLoss()
    if siamese_mode == 'triplet':
        criterion = tripletLoss(margin)
    if siamese_mode == 'contrastive':
        criterion == contrastiveLoss(margin)

    use_gpu = torch.cuda.is_available()
    if use_gpu:

        model.cuda()
        criterion = criterion.cuda()
        print 'GPUs: ', torch.cuda.device_count()

        ## multi-gpu in single machine, for forward() and backward()
        ## if DataParallel is used,
        ## we should use 'model.module.attribute' instead of 'model.attribute'
        # model = torch.nn.parallel.DataParallel(model, device_ids=[0, 1, 2])

    optimizer = optim.Adam([
        {'params': model.feature.parameters(), 'lr': Config.learning_rate1},
        {'params': model.classifier.parameters(), 'lr': Config.learning_rate2}
    ])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    iter_count = 1
    check_step = 100
    train_loss = 0.0
    check_loss = []
    check_score = []

    for e in range(Config.train_epoch):
        ## test
        model.eval()
        feature_map = torch.FloatTensor()
        for data in tqdm(test_dataloader):
            img, _  = data
            if use_gpu:
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            feature = model(img)
            feature_map = torch.cat((feature_map, feature.cpu().data), 0)
        feature_map = feature_map.numpy()
        feature_map = preprocessing.normalize(feature_map, norm='l2', axis=1)
        print "test size: ", feature_map.shape
        # np.save('feature/feature_{}.npy'.format(e), feature_map)
        NS_score = computeScore(feature_map)
        check_score.append(NS_score)
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'epoch: {}, NS Score: {:.4f}'.format(e, NS_score)

        ## train
        model.train()
        scheduler.step()
        if siamese_mode == 'rank':
            for data in train_dataloader:
                img0, img1, img2 = data
                label = Variable(torch.ones(img0.size(0),1)).float()
                if use_gpu:
                    img0 = Variable(img0).cuda()
                    img1 = Variable(img1).cuda()
                    img2 = Variable(img2).cuda()
                    label = label.cuda()
                else:
                    img0 = Variable(img0)
                    img1 = Variable(img1)
                    img2 = Variable(img2)

                ## RankNet loss
                out = model(img0, img1, img2)
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

        elif siamese_mode == 'triplet':
            for data in train_dataloader:
                img0, img1, img2 = data
                if use_gpu:
                    img0 = Variable(img0).cuda()
                    img1 = Variable(img1).cuda()
                    img2 = Variable(img2).cuda()
                else:
                    img0 = Variable(img0)
                    img1 = Variable(img1)
                    img2 = Variable(img2)

                ## Triplet loss
                anchor = model(img0)
                positive = model(img1)
                negative = model(img2)

                anchor = F.normalize(anchor, p=2, dim=1)
                positive = F.normalize(positive, p=2, dim=1)
                negative = F.normalize(negative, p=2, dim=1)

                loss = criterion(anchor, positive, negative)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]

                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1

        elif siamese_mode == 'contrastive':
            for data in train_dataloader:
                label, img1, img2 = data
                ## warning: the reshape for label if important to avoid producing matrix
                label = label.view(-1) # size: batch_size x 1 => batch_size, i.e., squeeze
                if use_gpu:
                    img1 = Variable(img1).cuda()
                    img2 = Variable(img2).cuda()
                    label = Variable(label).cuda()
                else:
                    img1 = Variable(img1)
                    img2 = Variable(img2)
                    label = Variable(label)

                ## contrastive loss
                output1 = model(img1)
                output2 = model(img2)
                output1 = F.normalize(output1, p=2, dim=1)
                output2 = F.normalize(output2, p=2, dim=1)

                loss = criterion(output1, output2, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]

                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1


    # np.save('result/train_loss_{}.npy'.format(time.strftime('%Y%m%d%H%M')), check_loss)
    # np.save('result/test_score_{}.npy'.format(time.strftime('%Y%m%d%H%M')), check_score)
    # torch.save(model, 'model/model_{}_{}_{}.pkl'.format(net_type, siamese_mode, time.strftime('%Y%m%d%H')))

    vis = visdom.Visdom(env='ukbench')
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
            X = np.arange(1, len(check_score)+1, 1),
            Y = np.array(check_score),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='epoch',
                ylabel='score'
            )
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Ukbench: 6200(training) + 4000(test)')
    parser.add_argument('-m', '--mode', type=str, required=False, help='siamese mode: rank(\'rank\', default), triplet(\'triplet\'), contrastive(\'contrastive\')')
    parser.add_argument('-n', '--net', type=str, required=False, help='CNN: AlexNet(\'A\', default), VGGNet(\'V\'), ResNet(\'R\')')
    parser.add_argument('-g', '--margin', type=float, required=False, help='margin for triplet loss and contrastive loss (default 0.1)')
    parser.set_defaults(mode='rank', net='A', margin=0.1)
    args = parser.parse_args()
    print 'Parameters: ', args

    train(siamese_mode=args.mode, net_type=args.net, margin=args.margin)
