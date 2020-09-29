#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torchvision
import csv
import numpy as np
import pandas as pd
from model import *
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, Linear, Conv2d
from data_generator import Generator
from tools import save_model, load_model, accuracy
from torch.nn.init import xavier_normal_, constant_
from inception import *

class Train():
    def __init__(self,
                 batch = 16,
                 lr = 0.01,
                 load_pretrain = False,
                 model = 1,
                 aug = False,
                 mixup = False
                 ):
        if model == 1:
            '''
            AlexNet+DSC
            5层卷积全变为深度可分离卷积层
            '''
            self.model = Net1()
            self.m = 1
        elif model == 2:
            self.model = Inception_FPN(5)
            self.m = 2
        elif model == 3:
            self.m = 3
            self.model = Classifier()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print('Lets use', torch.cuda.device_count(), 'GPUs!')
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.load_pretrain = load_pretrain
        self.optimizer = Adam(self.model.parameters(), lr = lr, weight_decay = 0.001)
        self.lambda1 = lambda epoch : 0.95**(epoch-20)
        self.lambda2 = lambda epoch : epoch/20
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda = self.lambda2)
        if self.load_pretrain == True:
            load_model(self.model, self.optimizer, self.scheduler, self.m)
            self.train_loss = ['train_loss']
            self.train_acc = ['train_acc']
            self.verify_loss = ['verify_loss']
            self.verify_acc = ['verify_acc']
            self.lr = ['learning rate']

            data = pd.read_csv('./train_data'+str(self.m)+'.csv')
            self.train_loss.extend(data['train_loss'].tolist())
            self.train_acc.extend(data['train_acc'].tolist())
            self.verify_loss.extend(data['verify_loss'].tolist())
            self.verify_acc.extend(data['verify_acc'].tolist())
            self.lr.extend(data['learning rate'].tolist())
        else:
            self.train_loss = ['train_loss']
            self.train_acc = ['train_acc']
            self.verify_loss = ['verify_loss']
            self.verify_acc = ['verify_acc']
            self.lr = ['learning rate']

            for i in self.model.parameters():
                if len(i.shape)>=2:
                    xavier_normal_(i)
            if self.m == 3:
                for i in self.model.named_modules():
                    if isinstance(i[1],CBLR):
                        constant_(i[1].BatchNorm2d.weight,0)
        self.mixup = mixup
        self.train_generator = Generator(batch = batch, mode = 'train',aug=aug)
        self.test_generator = Generator(batch = batch, mode = 'verify')

    def evaluate(self):
        self.model.eval()
        Loss = CrossEntropyLoss(weight=torch.Tensor([0.9,1,1])).to(self.device)
        loss = 0.0
        acc = 0.0
        with torch.no_grad():
            for i in range(10):
                images, labels = next(self.test_generator)
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss +=  Loss(logits, labels)
                acc += accuracy(logits, labels)
        self.model.train()
        return loss/10, acc/10
    
    def train(self, epoch_num = 10, step_one_epoch = 20, save_frq = 1000, evl_frq = 500):
        self.model.train()
        Loss = CrossEntropyLoss(weight=torch.Tensor([0.9,1,1])).to(self.device)
        if self.load_pretrain == True:
            global_step = int(os.listdir('./model' + str(self.m))[0])
            epoch = global_step//step_one_epoch
            _, max_acc = self.evaluate()
            self.train_loss = self.train_loss[0:global_step//evl_frq+1]
            self.train_acc = self.train_acc[0:global_step//evl_frq+1]
            self.verify_loss = self.verify_loss[0:global_step//evl_frq+1]
            self.verify_acc = self.verify_acc[0:global_step//evl_frq+1]
            self.lr = self.lr[0:global_step//evl_frq+1]
        else:
            global_step = 0
            epoch = 1
            max_acc = 0.0
        while(epoch<epoch_num):
            if(epoch>20):
                self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda = self.lambda1)
            self.scheduler.step(epoch)
            for i in range(step_one_epoch):
                if self.mixup == True:
                    images1, labels1 = next(self.train_generator)
                    images2, labels2 = next(self.train_generator)
                    lam = np.random.beta(0.4, 0.4)
                    images1 = images1.to(self.device)
                    images2 = images2.to(self.device)
                    images = images1*lam + images2*(1-lam)
                    images = images.to(self.device)
                    labels1 = labels1.to(self.device)
                    labels2 = labels2.to(self.device)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    loss = Loss(logits, labels1)*lam + Loss(logits, labels2)*(1-lam)
                    loss.backward()
                    self.optimizer.step()
                else:
                    images, labels = next(self.train_generator)
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    loss = Loss(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                if global_step%evl_frq==0:
                    if self.mixup == True:
                        with torch.no_grad():
                            logits1 = self.model(images1)
                            logits2 = self.model(images2)
                            train_loss = (Loss(logits1, labels1)+Loss(logits2, labels2))/2
                            train_acc = (accuracy(logits1, labels1)+accuracy(logits2, labels2))/2
                    else:
                        train_loss = loss
                        train_acc = accuracy(logits, labels)
                    verify_loss, verify_acc = self.evaluate()
                    print('step:     {:}     learning rate:     {:6f}'.format(global_step,self.scheduler.get_lr()[0]))
                    print('train_loss:     {:4f}     train_acc:     {:4f}'.format(train_loss, train_acc))
                    print('verify_loss:    {:4f}     verify_acc:    {:4f}'.format(verify_loss, verify_acc))
                    self.train_loss.append(float(train_loss))
                    self.train_acc.append(train_acc)
                    self.verify_loss.append(float(verify_loss))
                    self.verify_acc.append(verify_acc)
                    self.lr.append(float(self.scheduler.get_lr()[0]))
                    train_data = [self.lr,self.train_loss,self.train_acc,self.verify_loss,self.verify_acc]
                    train_data = np.array(train_data).T
                    with open('train_data'+str(self.m)+'.csv','w') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(train_data)
                    if global_step%save_frq==0:
                        if verify_acc >= max_acc:
                            save_model(self.model,self.optimizer,self.scheduler, global_step, self.m)
                            max_acc = verify_acc
                            if max_acc>0.92:
                                epoch=epoch_num
       
                global_step += 1
            epoch+=1
