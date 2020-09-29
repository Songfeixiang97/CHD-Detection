#!/usr/bin/env python
# coding: utf-8

import torch
import csv
import pandas as pd
import numpy as np
from tools import *
from model import *
from Dictionary import lab
from inception import *
from data_generator import Generator

model1 = Net1()
model2 = Inception(5)
model3 = Classifier()
if torch.cuda.device_count()>1:
    model1 = torch.nn.DataParallel(model1)
    model2 = torch.nn.DataParallel(model2)
    model3 = torch.nn.DataParallel(model3)
model1 = model1.to('cpu')
model2 = model2.to('cpu')
model3 = model3.to('cpu')
load_model(model1,m=1)
load_model(model2,m=2)
load_model(model3,m=3)
model1 = model1.eval()
model2 = model2.eval()
model3 = model3.eval()

true_labels = ['true_label']
pre_labels1 = ['AlexNet pre_label']
pre_labels2 = ['InceptionV4 pre_label']
pre_labels3 = ['YOLOV3 pre_label']
pre_labels4 = ['Sum pre_label']
acc1 = 0.0
acc2 = 0.0
acc3 = 0.0
acc4 = 0.0
with torch.no_grad():
    for i in range(30):
        a = Generator(32, mode = 'test')
        images, labels = next(a)
        images = images.to('cpu')
        labels = labels.to('cpu')
        logits1 = model1(images)
        logits2 = model2(images)
        logits3 = model3(images)
        logits4 = (logits1+logits2+logits3)/3.
        acc1 += accuracy(logits1, labels)
        acc2 += accuracy(logits2, labels)
        acc3 += accuracy(logits3, labels)
        acc4 += accuracy(logits4, labels)
        for t in labels:
            true_labels.append(lab[int(t)])
        for p1 in logits1.argmax(1):
            pre_labels1.append(lab[int(p1)])
        for p2 in logits2.argmax(1):
            pre_labels2.append(lab[int(p2)])
        for p3 in logits3.argmax(1):
            pre_labels3.append(lab[int(p3)])
        for p4 in logits4.argmax(1):
            pre_labels4.append(lab[int(p4)])
true_labels.append('model acc:')
pre_labels1.append(acc1/30)
pre_labels2.append(acc2/30)
pre_labels3.append(acc3/30)
pre_labels4.append(acc4/30)
data = [true_labels,pre_labels1,pre_labels2,pre_labels3,pre_labels4]
data = np.array(data).T
with open('test.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
print('model1 acc: ',acc1/30)
print('model2 acc: ',acc2/30)
print('model3 acc: ',acc3/30)
print('model4 acc: ',acc4/30)