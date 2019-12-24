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
from net import tripletLoss, contrastiveLoss, weight_init
from dataload import triTrainDataset, triSoftTrainDataset, contrasTrainDataset, triTrainDataset2
from dataset import buildTestData

class Config():
    ''' global parameters '''

    pair_num = 578 * 40
    batch_size = 32
    train_epoch = 30
    learning_rate1 = 2e-5
    learning_rate2 = 2e-5

    alpha = 0.1
    learning_rate3 = 2e-5

    img_transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_txt = '/path/to/train.txt'
    eval_func = '/path/to/compute_ap'
    retrieval_result = '/path/to/retrieval'
    test_dataset = {
        'oxf': {
            'img_testfolder': '/path/to/oxford5k',
            'img_testpath': '/path/to/oxford5k/images',
            'gt_path': '/path/to/oxford5k_groundTruth',
        },
        'par': {
            'img_testfolder': '/path/to/paris6k',
            'img_testpath': '/path/to/paris6k/images',
            'gt_path': '/path/to/paris6k_groundTruth',
        }
    }

    building_oxf = buildTestData(img_path=test_dataset['oxf']['img_testpath'], gt_path=test_dataset['oxf']['gt_path'], eval_func=eval_func)
    building_par = buildTestData(img_path=test_dataset['par']['img_testpath'], gt_path=test_dataset['par']['gt_path'], eval_func=eval_func)
    building = {
        'oxf': building_oxf,
        'par': building_par,
    }

    test_data_oxf = ImageFolder(test_dataset['oxf']['img_testfolder'], transform=img_transform)
    test_dataloader_oxf = DataLoader(dataset=test_data_oxf, shuffle=False, num_workers=4, batch_size=batch_size)
    test_data_par = ImageFolder(test_dataset['par']['img_testfolder'], transform=img_transform)
    test_dataloader_par = DataLoader(dataset=test_data_par, shuffle=False, num_workers=4, batch_size=batch_size)
    test_dataloader = {
        'oxf': test_dataloader_oxf,
        'par': test_dataloader_par,
    }

def train(siamese_mode, net_type, margin):

    if siamese_mode == 'rank':
        print "========== Siamese-Rank =========="
        train_data = triTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'rank2':
        print "========== Siamese-Rank2 =========="
        train_data = triTrainDataset2(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'tri':
        print "========== Siamese-Triplet =========="
        train_data = triTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'contra':
        print "========== Siamese-Contrastive =========="
        train_data = contrasTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    elif siamese_mode == 'rank-softmax' or siamese_mode == 'tri-softmax':
        print "========== Siamese-Rank-Softmax =========="
        train_data = triSoftTrainDataset(txt=Config.train_txt, \
            pair_num=Config.pair_num, transform=Config.img_transform)

    else:
        print "You should set siamese mode."
        # sys.exit(0)
        return

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4, batch_size=Config.batch_size)


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

    if siamese_mode == 'rank' or siamese_mode == 'rank2':
        criterion = nn.BCEWithLogitsLoss()
    if siamese_mode == 'tri':
        criterion = tripletLoss(margin)
    if siamese_mode == 'contra':
        criterion1 = contrastiveLoss(margin)
        criterion2 = nn.CrossEntropyLoss()
    if siamese_mode == 'rank-softmax':
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = nn.CrossEntropyLoss()
    if siamese_mode == 'tri-softmax':
        criterion1 = tripletLoss(margin)
        criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {'params': model.feature.parameters(), 'lr': Config.learning_rate1},
        {'params': model.classifier.parameters(), 'lr': Config.learning_rate2},
        {'params': model.fc.parameters(), 'lr': Config.learning_rate3}
    ])

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model.cuda()
        if siamese_mode == 'rank-softmax' or siamese_mode == 'contra' or siamese_mode == 'tri-softmax':
            criterion1 = criterion1.cuda()
            criterion2 = criterion2.cuda()
        else:
            criterion = criterion.cuda()

    iter_count = 1
    check_step = 100
    train_loss = 0.0
    check_loss = []
    check_accuracy_oxf = []
    check_accuracy_par = []

    for e in range(Config.train_epoch):

        ## test
        model.eval()
        for building_key in Config.building.keys():
            feature_map = torch.FloatTensor()
            for data in tqdm(Config.test_dataloader[building_key]):
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
            e_accuracy = Config.building[building_key].evalRetrieval(similarity, Config.retrieval_result)
            print time.strftime('%Y-%m-%d %H:%M:%S'), 'epoch: {}, mAP: {:.4f}'.format(e, e_accuracy)
            if building_key == 'oxf':
                check_accuracy_oxf.append(e_accuracy)
            if building_key == 'par':
                check_accuracy_par.append(e_accuracy)

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

                out = model(anchor, positive, negative)

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

        elif siamese_mode == 'rank2':
            for data in train_dataloader:
                anchor, positive, candidate, label = data
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    candidate = Variable(candidate).cuda()
                    label = Variable(label).cuda()
                else:
                    anchor = Variable(anchor)
                    positive = Variable(positive)
                    candidate = Variable(candidate)
                    label = Variabel(label)

                out = model(anchor, positive, candidate)

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
                target, label_1, label_2, img1, img2 = data
                ## warning: the reshape for label if important to avoid producing matrix
                target = target.view(-1) # size: batch_size x 1 => batch_size
                label_1 = label_1.view(-1)
                label_2 = label_2.view(-1)
                if use_gpu:
                    img1 = Variable(img1).cuda()
                    img2 = Variable(img2).cuda()
                    target = Variable(target).cuda()
                    label_1 = Variable(label_1).cuda()
                    label_2 = Variable(label_2).cuda()
                else:
                    img1 = Variable(img1)
                    img2 = Variable(img2)
                    target = Variable(target)
                    label_1 = Variable(label_1)
                    label_2 = Variable(label_2)

                ## contrastive loss
                output1 = model(img1)
                output2 = model(img2)

                # z1 = model.fc(output1)
                # z2 = model.fc(output2)

                output1 = F.normalize(output1, p=2, dim=1)
                output2 = F.normalize(output2, p=2, dim=1)

                # loss_class = criterion2(z1, label_1) + criterion2(z2, label_2)
                # loss = loss_class + Config.alpha * criterion1(output1, output2, target)
                loss = criterion1(output1, output2, target)

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

                loss1 = criterion1(o, rank_label)
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
                anchor_label = anchor_label.view(-1)
                positive_label = positive_label.view(-1)
                negative_label = negative_label.view(-1)
                if use_gpu:
                    anchor = Variable(anchor).cuda()
                    positive = Variable(positive).cuda()
                    negative = Variable(negative).cuda()
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

                anchor_feature = model(anchor)
                positive_feature = model(positive)
                negative_feature = model(negative)

                z1 = model.fc(positive_feature)
                z2 = model.fc(negative_feature)

                anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
                positive_feature = F.normalize(positive_feature, p=2, dim=1)
                negative_feature = F.normalize(negative_feature, p=2, dim=1)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dataset: Paris6k, Oxford5k')
    parser.add_argument('-m', '--mode', type=str, required=False, help='siamese mode: rank(\'rank\', default), rank2(\'rank2\'), triplet(\'tri\'), contrastive(\'contra\'), rank-softmax(\'rank-softmax\')')
    parser.add_argument('-n', '--net', type=str, required=False, help='CNN: AlexNet(\'A\', default), VGGNet(\'V\'), ResNet(\'R\')')
    parser.add_argument('-g', '--margin', type=float, required=False, help='margin for triplet loss and contrastive loss (default 0.1)')
    parser.set_defaults(mode='rank', net='A', margin=0.1)
    args = parser.parse_args()
    print 'Parameters: ', args

    train(siamese_mode=args.mode, net_type=args.net, margin=args.margin)
