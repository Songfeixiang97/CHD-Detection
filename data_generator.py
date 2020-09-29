#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import torch
import os
import random
from skimage import exposure
from data_augment import data_aug
import numpy as np


class Generator():
    def __init__(self,
                 batch = 128,
                 mode = 'train',
                 aug = False
                ):
        self.aug = aug
        self.batch = batch
        self.index = 0
        self.image_files, self.labels = self.read_file(mode)
        self.len = len(self.image_files)
        self.all = np.arange(self.len)
        random.shuffle(self.all)
        self.mode = mode
        
    def __iter__(self):
        return
        
    def read_file(self, mode):
        paths = ['./dataset5CRS/'+mode+'/Positive', './dataset5CRS/'+mode+'/Negative/ASD', './dataset5CRS/'+mode+'/Negative/VSD']
        filename1 = os.listdir(paths[0])
        filename2 = os.listdir(paths[1])
        filename3 = os.listdir(paths[2])
        image_files = []
        labels = []
        k = 0 
        for i in [filename1, filename2, filename3]:
            for j in i:
                image_file = os.path.join(paths[k],j)
                image_files.append(image_file)
                labels.append(k)
            k+=1
        return image_files, labels
    
    def __next__(self):
        if self.index+self.batch>=self.len:
            self.index = 0
            random.shuffle(self.all)
        images = []
        labels = []
        if self.mode=='train':
           # size = random.choice([256,288,224,192,160,128])
            size = 224
        else:
            size = 224
        dd = 1/size
        if self.mode=='train' and self.aug==True:
            image = []
            for i in self.all[self.index:self.index+self.batch]:
                img_file = self.image_files[i]
                imgs_filename = os.listdir(img_file)
                imgs_filename.sort()
                for j in imgs_filename:
                    #print(img_file,j)
                    img = cv.imread(img_file+'/'+j)
                    img = cv.resize(img,(size,size))
                    image.append(img)
                labels.append(self.labels[i])
            image = np.array(image)
            image = data_aug(image)
            for b in range(self.batch):
                gray_imgs = []
                for i in image[b*5:b*5+5]:
                    img = cv.cvtColor(i,cv.COLOR_BGR2GRAY)
                    img = (img-img.mean())/max(img.std(),dd)
                    gray_imgs.append(img)
                images.append(gray_imgs)  
            
        else:
            for i in self.all[self.index:self.index+self.batch]:
                img_file = self.image_files[i]
                imgs_filename = os.listdir(img_file)
                imgs_filename.sort()
                image = []
                for j in imgs_filename:
                    #print(img_file,j)
                    img = cv.imread(img_file+'/'+j, cv.IMREAD_GRAYSCALE)
                    img = cv.resize(img,(size,size))
                    img = (img-img.mean())/max(img.std(),dd)
                    image.append(img)
                images.append(image)
                labels.append(self.labels[i])
            #########################################
        self.index+=self.batch
        images = np.array(images)
        images = torch.Tensor(images)
        labels = torch.LongTensor(labels)
        return images, labels