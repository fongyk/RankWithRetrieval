import torch
from torch.utils.data import Dataset

import os
import sys
import linecache
import random
import numpy as np
from PIL import Image
from skimage import io

## siamese-triplet for training
class triTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.building_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.building_num

    def __getitem__(self, index):
        anchor = linecache.getline(self.txt, (index % self.building_num) + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_img = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_img = anchor + '/' + anchor_folder[positive_id]
                break

        while True:
            negative = linecache.getline(self.txt, random.randint(1, self.building_num)).strip('\n')
            if negative != anchor:
                negative_folder = os.listdir(negative)
                negative_id = random.randint(0, len(negative_folder)-1)
                negative_img = negative + '/' + negative_folder[negative_id]
                break


        anchor_img = Image.open(anchor_img).convert('RGB')
        positive_img = Image.open(positive_img).convert('RGB')
        negative_img = Image.open(negative_img).convert('RGB')


        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return self.pair_num

## siamese-triplet for training, the candidate can be positive or negative
class triTrainDataset2(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.building_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.building_num

    def __getitem__(self, index):
        anchor = linecache.getline(self.txt, (index % self.building_num) + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_img = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_img = anchor + '/' + anchor_folder[positive_id]
                break

        r = random.randint(0, 9)
        if r <= 7:
            label = torch.FloatTensor([1.0])
            while True:
                candidate = linecache.getline(self.txt, random.randint(1, self.building_num)).strip('\n')
                if candidate != anchor:
                    candidate_folder = os.listdir(candidate)
                    candidate_id = random.randint(0, len(candidate_folder)-1)
                    candidate_img = candidate + '/' + candidate_folder[candidate_id]
                    break
        else:
            label = torch.FloatTensor([0.5])
            while True:
                candidate_id = random.randint(0, len(anchor_folder)-1)
                if candidate_id != anchor_id and candidate_id != positive_id:
                    candidate_img = anchor + '/' + anchor_folder[candidate_id]
                    break

        anchor_img = Image.open(anchor_img).convert('RGB')
        positive_img = Image.open(positive_img).convert('RGB')
        candidate_img = Image.open(candidate_img).convert('RGB')

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            candidate_img = self.transform(candidate_img)

        return anchor_img, positive_img, candidate_img, label

    def __len__(self):
        return self.pair_num

## siamese-triplet + softmax for training
class triSoftTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.building_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.building_num

    def __getitem__(self, index):
        anchor_label = index % self.building_num
        anchor = linecache.getline(self.txt, anchor_label + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_img = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_img = anchor + '/' + anchor_folder[positive_id]
                break

        positive_label = anchor_label

        while True:
            negative_label = random.randint(0, self.building_num - 1)
            negative = linecache.getline(self.txt, negative_label + 1).strip('\n')
            if negative != anchor:
                negative_folder = os.listdir(negative)
                negative_id = random.randint(0, len(negative_folder)-1)
                negative_img = negative + '/' + negative_folder[negative_id]
                break

        anchor_img = Image.open(anchor_img).convert('RGB')
        positive_img = Image.open(positive_img).convert('RGB')
        negative_img = Image.open(negative_img).convert('RGB')


        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, \
            torch.from_numpy(np.array([anchor_label])),\
            torch.from_numpy(np.array([positive_label])),\
            torch.from_numpy(np.array([negative_label]))

    def __len__(self):
        return self.pair_num

## siamese-contrastive for training
class contrasTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.building_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.building_num

    def __getitem__(self, index):
        img0 = linecache.getline(self.txt, (index % self.building_num) + 1).strip('\n')
        img0_folder = os.listdir(img0)
        img0_id = random.randint(0, len(img0_folder)-1)
        img0_img = img0 + '/' + img0_folder[img0_id]

        label = random.randint(0, 1)
        if label==1:
            while True:
                img1_id = random.randint(0, len(img0_folder)-1)
                if img1_id != img0_id:
                    img1_img = img0 + '/' + img0_folder[img1_id]
                    break
        else:
            while True:
                img1 = linecache.getline(self.txt, random.randint(1, self.building_num)).strip('\n')
                if img1 != img0:
                    img1_folder = os.listdir(img1)
                    img1_id = random.randint(0, len(img1_folder)-1)
                    img1_img = img1 + '/' + img1_folder[img1_id]
                    break

        img0_img = Image.open(img0_img).convert('RGB')
        img1_img = Image.open(img1_img).convert('RGB')

        if self.transform is not None:
            img0_img = self.transform(img0_img)
            img1_img = self.transform(img1_img)

        return torch.from_numpy(np.array([label], dtype=np.float32)), img0_img, img1_img

    def __len__(self):
        return self.pair_num


## N-pair siamese-triplet for training
class triNpTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.building_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.building_num

    def __getitem__(self, index):
        anchor = linecache.getline(self.txt, (index % self.building_num) + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_img = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_img = anchor + '/' + anchor_folder[positive_id]
                break

        anchor_img = Image.open(anchor_img).convert('RGB')
        positive_img = Image.open(positive_img).convert('RGB')

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)

        return anchor_img, positive_img, torch.from_numpy(np.array([index % self.building_num])).float()

    def __len__(self):
        return self.pair_num
