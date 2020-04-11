import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import folder

import os
import linecache
import random
from PIL import Image
import numpy as np
import sys

## make dataset
def makeDataset(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]:int(classes[i]) for i in range(len(classes))}
    # samples = folder.make_dataset(dir, class_to_idx)
    samples = folder.make_dataset(dir, class_to_idx, extensions=['.jpeg', '.jpg'])
    return samples

## siamese-contrastive
class myContrasDataset(Dataset):
    def __init__(self, img_rootpath, txt, transform=None):
        self.rootpath = img_rootpath
        self.transform = transform
        self.txt = txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))
        line.strip('\n')
        img0_list = line.split()
        group_id = int(img0_list[1])
        group_path = self.rootpath + '/{:04d}'.format(group_id)
        img0_group = os.listdir(group_path)
        while True:
            img1_id = random.randint(0, len(img0_group)-1)
            if img0_group[img1_id] != (img0_list[0].split('/'))[-1]:
                img1_path = group_path + '/' + img0_group[img1_id]
                break
        while True:
            img2_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
            if img2_list[1] != img0_list[1]:
                break

        label = random.randint(0, 1)
        sample_1 = img0_list[0]
        if label==0:
            sample_2 = img2_list[0]
        else:
            sample_2 = img1_path

        sample_1 = Image.open(sample_1).convert('RGB')
        sample_2 = Image.open(sample_2).convert('RGB')

        if self.transform is not None:
            sample_1 = self.transform(sample_1)
            sample_2 = self.transform(sample_2)

        return torch.from_numpy(np.array([label], dtype=np.float32)), sample_1, sample_2

    def __len__(self):
        fr = open(self.txt, 'r')
        num = len(fr.readlines())
        fr.close()
        return num

## siamese-triplet
class myTriDataset_anchor(Dataset):
    def __init__(self, img_rootpath, txt, anchorDir, pair_num=None, transform=None):
        self.rootpath = img_rootpath
        self.transform = transform
        self.txt = txt
        self.anchors = makeDataset(anchorDir)
        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = len(self.anchors)

        fr = open(self.txt, 'r')
        self.img_num = len(fr.readlines())
        fr.close()

    def __getitem__(self, index):
        img0_path, group_id = self.anchors[index % len(self.anchors)]
        group_path = self.rootpath + '/{:04d}'.format(group_id)
        img0_group = os.listdir(group_path)
        while True:
            img1_id = random.randint(0, len(img0_group)-1)
            if img0_group[img1_id] != (img0_path.split('/'))[-1]:
                img1_path = group_path + '/' + img0_group[img1_id]
                break
        while True:
            img2_list = linecache.getline(self.txt, random.randint(1, self.img_num)).strip('\n').split()
            if (img2_list[1].split('/'))[-1] != (img0_path.split('/'))[-1]:
                break

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_list[0]).convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        return self.pair_num

## siamese-triplet
class myTriDataset_random(Dataset):
    def __init__(self, img_rootpath, txt, transform=None):
        self.rootpath = img_rootpath
        self.transform = transform
        self.txt = txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))
        line.strip('\n')
        img0_list = line.split()
        group_id = int(img0_list[1])
        group_path = self.rootpath + '/{:04d}'.format(group_id)
        img0_group = os.listdir(group_path)
        while True:
            img1_id = random.randint(0, len(img0_group)-1)
            if img0_group[img1_id] != (img0_list[0].split('/'))[-1]:
                img1_path = group_path + '/' + img0_group[img1_id]
                break
        while True:
            img2_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
            if img2_list[1] != img0_list[1]:
                break

        img0 = Image.open(img0_list[0]).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_list[0]).convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        fr = open(self.txt, 'r')
        num = len(fr.readlines())
        fr.close()
        return num
