import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
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
from net import tripletLoss, contrastiveLoss, NpairLoss, weight_init
from dataload import triTrainDataset, triSoftTrainDataset, contrasTrainDataset, testDataset

class Config():
    ''' global parameters '''

    pair_num = 10575 * 5

    img_trainpath = '/path/to/CASIA_WebFace'
    train_txt = '/path/to/train.txt'
    img_testpath = '/path/to/lfw'
    testpairs_txt = '/path/to/pairs.txt'
    testpeople_txt = '/path/to/people.txt'

    img_transform = transforms.Compose([
        ## warning: if the dataloader is not customized,
        ## images in one batch shoube be of the same size.

        ## Scale(l): smaller edge of image will be matched to l
        transforms.Resize((100, 100)), ## RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # transforms.Resize(100), ## Gray
        # transforms.Grayscale(3),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5,], std=[0.5,])
    ])

    batch_size = 64
    train_epoch = 30
    learning_rate1 = 5e-5
    learning_rate2 = 1e-4
    beta = 1.0

    alpha = 0.0001
    learning_rate3 = 1e-4

def writeCSV(dataDict, fileName):
    ''' write a dict to a csv file'''

    with open(fileName, "w") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in dataDict.iteritems():
            csvWriter.writerow([k,v])
        csvFile.close()

def evalIdentification(feature_map, pairs_txt, threshold):
    ''' compute the face identification accuracy '''

    with open(pairs_txt) as fr:
        lines = fr.readlines()
        pair_num = len(lines) - 1 ## 6000
        fold, _ = lines[0].strip('\n').split('\t')
        fold = int(fold)
        pair_fold = pair_num / fold ## 600
        accuracy = np.zeros((fold, pair_fold), dtype=float)
        for i in range(1, pair_num):
            line = lines[i].strip('\n').split('\t')
            if len(line) == 3:
                person_name = line[0]
                id_1 = int(line[1])
                id_2 = int(line[2])
                cos_sim = np.sum(feature_map[person_name][id_1] * feature_map[person_name][id_2])
                x = (i-1) / pair_fold
                y = (i-1) % pair_fold
                if cos_sim >= threshold:
                    accuracy[x, y] = 1.0
            if len(line) == 4:
                person_1 = line[0]
                id_1 = int(line[1])
                person_2 = line[2]
                id_2 = int(line[3])
                cos_sim = np.sum(feature_map[person_1][id_1] * feature_map[person_2][id_2])
                x = (i-1) / pair_fold
                y = (i-1) % pair_fold
                if cos_sim < threshold:
                    accuracy[x, y] = 1.0
            # print cos_sim
        score = np.sum(accuracy, axis=1) / pair_fold
        total_accuracy = np.mean(score)
        std_err = np.std(score, ddof=1)

        return total_accuracy, std_err


def train(siamese_mode, net_type, threshold, margin):

    if siamese_mode == 'rank':
        print "========== Siamese-Rank =========="
        train_data = triTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'tri':
        print "========== Siamese-Triplet =========="
        train_data = triTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'contra':
        print "========== Siamese-Contrastive =========="
        train_data = contrasTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'rank-softmax':
        print "========== Siamese-Rank-Softmax =========="
        train_data = triSoftTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'tri-softmax':
        print "========== Siamese-Triplet-Softmax =========="
        train_data = triSoftTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    else:
        print "You should set siamese mode."
        # sys.exit(0)
        return

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4, batch_size=Config.batch_size)

    test_data = testDataset(dir=Config.img_testpath, transform=Config.img_transform)
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
    if net_type == 'F':
        model = net.faceNet()
        model.apply(weight_init)
        print "---------- faceNet ---------"

    ## weights initialization for traning from scratch
    # model.apply(weight_init)

    nploss =  NpairLoss()

    if siamese_mode == 'rank':
        criterion = nn.BCEWithLogitsLoss()
    if siamese_mode == 'tri':
        criterion = tripletLoss(margin)
    if siamese_mode == 'contra':
        criterion = contrastiveLoss(margin)
    if siamese_mode == 'rank-softmax':
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = nn.CrossEntropyLoss()
    if siamese_mode == 'tri-softmax':
        criterion1 = tripletLoss(margin)
        criterion2 = nn.CrossEntropyLoss()

    if siamese_mode == 'rank-softmax' or siamese_mode == 'tri-softmax':
        optimizer = optim.Adam([
            {'params': model.feature.parameters(), 'lr': Config.learning_rate1},
            {'params': model.classifier.parameters(), 'lr': Config.learning_rate2},
            {'params': model.fc.parameters(), 'lr': Config.learning_rate3, 'weight_decay': 5e-4}
        ])
    else:
        optimizer = optim.Adam([
            {'params': model.feature.parameters(), 'lr': Config.learning_rate1},
            {'params': model.classifier.parameters(), 'lr': Config.learning_rate2}
        ])

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        if siamese_mode == 'rank-softmax' or siamese_mode == 'tri-softmax':
            criterion1 = criterion1.cuda()
            criterion2 = criterion2.cuda()
        else:
            criterion = criterion.cuda()

    iter_count = 1
    check_step = 100
    train_loss = 0.0
    check_loss = []
    check_accuracy = []
    check_std = []

    for e in range(Config.train_epoch):

        ## test
        model.eval()
        feature_map = {}
        for data in tqdm(test_dataloader):
            img, label = data
            if use_gpu:
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            feature = model(img)
            feature = F.normalize(feature, p=2, dim=1).cpu().data.numpy()
            person_names, idx = label
            for p in range(len(person_names)):
                if person_names[p] not in feature_map:
                    feature_map[person_names[p]] = {}
                feature_map[person_names[p]][idx[p]] = feature[p,:]
        e_accuracy = 0.0
        thrange = np.arange(0, 1.01, 0.01)
        for th in thrange:
            t_accuracy, t_std = evalIdentification(feature_map, Config.testpairs_txt, th)
            if t_accuracy > e_accuracy:
                mx_th = th
                std_err = t_std
                e_accuracy = t_accuracy
        check_accuracy.append(e_accuracy)
        check_std.append(std_err)
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'epoch: {}, th: {:.2f}, acc: {:.4f}, std: {:.4f}'.format(e, mx_th, e_accuracy, std_err)
       
        ## train
        model.train()
        scheduler.step()
        if siamese_mode == 'rank':
            for data in train_dataloader:
                anchor, positive, negative = data
                label = Variable(torch.ones(anchor.size(0), 1)).float()
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    negative = Variable(negative).cuda()
                    label = label.cuda()
                else:
                    anchor = Variable(anchor)
                    positive = Variable(positive)
                    negative = Variable(negative)

                o1, o2, x0, x1, x2 = model(anchor, positive, negative)

                loss = criterion(o2 - o1, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]
                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1

        elif siamese_mode == 'tri':
            for data in train_dataloader:
                anchor, positive, negative = data
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    negative = Variable(negative).cuda()
                else:
                    anchor = Variable(anchor)
                    positive = Variable(positive)
                    negative = Variable(negative)

                anchor_feature = model(anchor)
                positive_feature = model(positive)
                negative_feature = model(negative)

                anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
                positive_feature = F.normalize(positive_feature, p=2, dim=1)
                negative_feature = F.normalize(negative_feature, p=2, dim=1)

                loss = criterion(anchor_feature, positive_feature, negative_feature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]
                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1

        elif siamese_mode == 'contra':
            for data in train_dataloader:
                label, img1, img2 = data
                ## warning: the reshape for label if important to avoid producing matrix
                label = label.view(-1) # size: batch_size x 1 => batch_size
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

        elif siamese_mode == 'rank-softmax':
            for data in train_dataloader:
                anchor, positive, negative, anchor_label, positive_label, negative_label = data
                rank_label = Variable(torch.ones(anchor.size(0), 1)).float()
                anchor_label = anchor_label.view(-1)
                positive_label = positive_label.view(-1)
                negative_label = negative_label.view(-1)
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    negative = Variable(negative).cuda()
                    rank_label = rank_label.cuda()
                    anchor_label = Variable(anchor_label).cuda()
                    positive_label = Variable(positive_label).cuda()
                    negative_label = Variable(negative_label).cuda()
                else:
                    anchor = Variable(anchor)
                    positive = Variable(positive)
                    negative = Variable(negative)
                    anchor_label = Variable(anchor_label)
                    positive_label = Variable(positive_label)
                    negative_label = Variable(negative_label)

                o, z0, z1, z2 = model.forward_rank_softmax(anchor, positive, negative)

                # loss1 = criterion1(o, rank_label)
                o1, o2, x0, x1, x2 = model(anchor, positive, negative)
                x0n = F.normalize(x0, p=2, dim=1)
                x1n = F.normalize(x1, p=2, dim=1)
                x2n = F.normalize(x2, p=2, dim=1)
                loss1 = F.relu(1.0 - (x2n-x0n).pow(2).sum(1) * o2 + (x1n-x0n).pow(2).sum(1) * o1).mean()

                loss2 = criterion2(z1, positive_label) + criterion2(z2, negative_label)
                loss = loss1 + Config.alpha * loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]
                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1

        elif siamese_mode == 'tri-softmax':
            for data in train_dataloader:
                anchor, positive, negative, anchor_label, positive_label, negative_label = data
                rank_label = Variable(torch.ones(anchor.size(0), 1)).float()
                anchor_label = anchor_label.view(-1)
                positive_label = positive_label.view(-1)
                negative_label = negative_label.view(-1)
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    negative = Variable(negative).cuda()
                    rank_label = rank_label.cuda()
                    anchor_label = Variable(anchor_label).cuda()
                    positive_label = Variable(positive_label).cuda()
                    negative_label = Variable(negative_label).cuda()
                else:
                    anchor = Variable(anchor)
                    positive = Variable(positive)
                    negative = Variable(negative)
                    anchor_label = Variable(anchor_label)
                    positive_label = Variable(positive_label)
                    negative_label = Variable(negative_label)

                ## N-pair Loss
                # anchor_feature = model(anchor)
                # positive_feature = model(positive)
                # loss = nploss(anchor_feature, positive_feature, anchor_label)

                anchor_feature = model(anchor)
                positive_feature = model(positive)
                negative_feature = model(negative)
                anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
                positive_feature = F.normalize(positive_feature, p=2, dim=1)
                negative_feature = F.normalize(negative_feature, p=2, dim=1)

                # z0 = model.forward_classify(anchor)
                z1 = model.forward_classify(positive)
                z2 = model.forward_classify(negative)

                loss1 = criterion1(anchor_feature, positive_feature, negative_feature)
                loss2 = (criterion2(z1, positive_label) + criterion2(z2, negative_label))/2.0
                loss = loss1 + Config.alpha * loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data[0]
                if iter_count % check_step == 0:
                    check_loss.append(train_loss/check_step)
                    print time.strftime('%Y-%m-%d %H:%M:%S'), 'iter: {}, loss: {:.8f}'.format(iter_count, train_loss/check_step)
                    train_loss = 0.0

                iter_count += 1

    vis = visdom.Visdom(env='face')
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
                ylabel='score'
            )
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training Dataset: CASIA-WebFace, Test Dataset: Labeled Face in the Wild')
    parser.add_argument('-m', '--mode', type=str, required=False, help='siamese mode: rank(\'rank\', default), triplet(\'tri\'), contrastive(\'contra\'), rank-softmax(\'rank-softmax\'), triplet-softmax(\'tri-softmax\')')
    parser.add_argument('-n', '--net', type=str, required=False, help='CNN: AlexNet(\'A\', default), VGGNet(\'V\'), ResNet(\'R\'), faceNet(\'F\')')
    parser.add_argument('-t', '--threshold', type=float, required=False, help='threshold for computing accuracy (default 0.87)')
    parser.add_argument('-g', '--margin', type=float, required=False, help='margin for triplet loss and contrastive loss (default 0.1)')
    parser.set_defaults(mode='rank', net='A', threshold=0.87, margin=0.1)
    args = parser.parse_args()

    print 'Parameters: ', args

    ## threshold
    ## AlexNet + Rank: 0.87 : accuracy = 0.66 -> 0.80
    ## AlexNet + Triplet:
    ## VGGNet + Rank: 0.79 : accuracy = 0.65
    ## ResNet + Rank: 0.85 : accuracy = 0.68

    train(siamese_mode=args.mode, net_type=args.net, threshold=args.threshold, margin=args.margin)
