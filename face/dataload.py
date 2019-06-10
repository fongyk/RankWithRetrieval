import torch
from torch.utils.data import Dataset

import os
import sys
import linecache
import random
import numpy as np
from PIL import Image
from skimage import io

## make dataset for test
def makeDataset(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    samples = []
    for d in classes:
        d_path = dir + '/' + d
        sample_d = os.listdir(d_path)
        sample_d.sort()
        for i in range(len(sample_d)):
            ## id: starts from 1
            samples.append((d_path + '/' + sample_d[i], (d, i+1)))
    return samples

## siamese-triplet for training
class triTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.person_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.person_num

    def __getitem__(self, index):
        anchor = linecache.getline(self.txt, (index % self.person_num) + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_face = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_face = anchor + '/' + anchor_folder[positive_id]
                break

        while True:
            negative = linecache.getline(self.txt, random.randint(1, self.person_num)).strip('\n')
            if negative != anchor:
                negative_folder = os.listdir(negative)
                negative_id = random.randint(0, len(negative_folder)-1)
                negative_face = negative + '/' + negative_folder[negative_id]
                break


        anchor_face = Image.open(anchor_face)
        positive_face = Image.open(positive_face)
        negative_face = Image.open(negative_face)

        if anchor_face.mode != "RGB":
            origin = anchor_face
            anchor_face = Image.new("RGB", origin.size)
            anchor_face.paste(origin)
        if positive_face.mode != "RGB":
            origin = positive_face
            positive_face = Image.new("RGB", origin.size)
            positive_face.paste(origin)
        if negative_face.mode != "RGB":
            origin = negative_face
            negative_face = Image.new("RGB", origin.size)
            negative_face.paste(origin)

        # if anchor_face.mode == "RGB":
        #     anchor_face = anchor_face.convert('L')
        # if positive_face.mode == "RGB":
        #     positive_face = positive.convert('L')
        # if negative_face.mode == "RGB":
        #     negative_face = negative_face.convert('L')

        if self.transform is not None:
            anchor_face = self.transform(anchor_face)
            positive_face = self.transform(positive_face)
            negative_face = self.transform(negative_face)

        return anchor_face, positive_face, negative_face

    def __len__(self):
        return self.pair_num

## siamese-triplet + softmax for training
class triSoftTrainDataset(Dataset):
    def __init__(self, txt, pair_num=None, transform=None):
        self.txt = txt
        self.transform = transform

        fr = open(self.txt, 'r')
        self.person_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.person_num

    def __getitem__(self, index):
        anchor_label = index % self.person_num
        anchor = linecache.getline(self.txt, anchor_label + 1).strip('\n')
        anchor_folder = os.listdir(anchor)
        anchor_id = random.randint(0, len(anchor_folder)-1)
        anchor_face = anchor + '/' + anchor_folder[anchor_id]
        while True:
            positive_id = random.randint(0, len(anchor_folder)-1)
            if positive_id != anchor_id:
                positive_face = anchor + '/' + anchor_folder[positive_id]
                break

        positive_label = anchor_label

        while True:
            negative_label = random.randint(0, self.person_num - 1)
            negative = linecache.getline(self.txt, negative_label + 1).strip('\n')
            if negative != anchor:
                negative_folder = os.listdir(negative)
                negative_id = random.randint(0, len(negative_folder)-1)
                negative_face = negative + '/' + negative_folder[negative_id]
                break

        anchor_face = Image.open(anchor_face)
        positive_face = Image.open(positive_face)
        negative_face = Image.open(negative_face)

        if anchor_face.mode != "RGB":
            origin = anchor_face
            anchor_face = Image.new("RGB", origin.size)
            anchor_face.paste(origin)
        if positive_face.mode != "RGB":
            origin = positive_face
            positive_face = Image.new("RGB", origin.size)
            positive_face.paste(origin)
        if negative_face.mode != "RGB":
            origin = negative_face
            negative_face = Image.new("RGB", origin.size)
            negative_face.paste(origin)

        if self.transform is not None:
            anchor_face = self.transform(anchor_face)
            positive_face = self.transform(positive_face)
            negative_face = self.transform(negative_face)

        return anchor_face, positive_face, negative_face, \
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
        self.person_num = len(fr.readlines())
        fr.close()

        if pair_num is not None:
            self.pair_num = pair_num
        else:
            self.pair_num = self.person_num

    def __getitem__(self, index):
        img0 = linecache.getline(self.txt, (index % self.person_num) + 1).strip('\n')
        img0_folder = os.listdir(img0)
        img0_id = random.randint(0, len(img0_folder)-1)
        img0_face = img0 + '/' + img0_folder[img0_id]

        label = random.randint(0, 1)
        if label==1:
            while True:
                img1_id = random.randint(0, len(img0_folder)-1)
                if img1_id != img0_id:
                    img1_face = img0 + '/' + img0_folder[img1_id]
                    break
        else:
            while True:
                img1 = linecache.getline(self.txt, random.randint(1, self.person_num)).strip('\n')
                if img1 != img0:
                    img1_folder = os.listdir(img1)
                    img1_id = random.randint(0, len(img1_folder)-1)
                    img1_face = img1 + '/' + img1_folder[img1_id]
                    break

        img0_face = Image.open(img0_face)
        img1_face = Image.open(img1_face)

        if img0_face.mode != "RGB":
            origin = img0_face
            img0_face = Image.new("RGB", origin.size)
            img0_face.paste(origin)
        if img1_face.mode != "RGB":
            origin = img1_face
            img1_face = Image.new("RGB", origin.size)
            img1_face.paste(origin)

        if self.transform is not None:
            img0_face = self.transform(img0_face)
            img1_face = self.transform(img1_face)

        return torch.from_numpy(np.array([label])), img0_face, img1_face

    def __len__(self):
        return self.pair_num

## loading test data and label
class testDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.samples = makeDataset(dir)

    def __getitem__(self, index):
        face_path, label = self.samples[index]
        face_test = Image.open(face_path)

        if face_test.mode != "RGB":
            origin = face_test
            face_test = Image.new("RGB", origin.size)
            face_test.paste(origin)
            
        if self.transform is not None:
            face_test = self.transform(face_test)
            
        return face_test, label

    def __len__(self):
        return len(self.samples)
