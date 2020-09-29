#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import *

class CBL(nn.Module):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride = 1,
                 padding = 0
                 ):
        super(CBL, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.BatchNorm2d = BatchNorm2d(out_channels)
        self.ReLU = ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BatchNorm2d(x)
        x = self.ReLU(x)
        return x
    
class CBAM(nn.Module):
    def __init__(self,
                 in_channels
                ):
        super(CBAM,self).__init__()
        self.in_channels = in_channels
        self.avgpool = AdaptiveAvgPool2d(output_size = (1,1))
        self.maxpool = AdaptiveMaxPool2d(output_size = (1,1))
        self.linear = nn.Sequential(
            Linear(in_channels,in_channels//2),
            Linear(in_channels//2,in_channels)
        )
        self.relu = ReLU(inplace = True)
        self.avgpool1d = AdaptiveAvgPool1d(output_size = 1)
        self.maxpool1d = AdaptiveMaxPool1d(output_size = 1)
        self.conv = Conv2d(2,1,3,1,1)
        self.sigmoid = Sigmoid()
    def forward(self,x):
        x1 = self.avgpool(x).squeeze(2).squeeze(2)
        x2 = self.maxpool(x).squeeze(2).squeeze(2)
        x3 = self.linear(x1)
        x4 = self.linear(x2)
        x5 = self.relu(x3+x4).unsqueeze(-1).unsqueeze(-1)
        x6 = self.in_channels*x5/x5.sum(1).unsqueeze(-1)
        y = torch.mul(x,x6)
        y1 = self.avgpool1d(y.view(x.shape[0],-1,x.shape[1]))
        y2 = self.maxpool1d(y.view(x.shape[0],-1,x.shape[1]))
        y3 = torch.cat((y1,y2),dim = 2).permute(0,2,1).view(x.shape[0],2,x.shape[2],x.shape[3])
        y4 = self.conv(y3)
        y5 = self.sigmoid(y4)
        a = y5.view(x.shape[0],-1).sum(1)
        y6 = x.shape[2]*x.shape[3]*y5/a.view(x.shape[0],1,1,1)
        y = torch.mul(y,y6)
        return y

class Stem(nn.Module):
    def __init__(self,
                 in_channels):
        super(Stem,self).__init__()
        self.Conv1 = nn.Sequential(
            CBL(in_channels,32,3,2),
            CBL(32,64,3,1,1)
        )
        self.maxpool2_1 = MaxPool2d(3,2)
        self.Conv2_2 = CBL(64,64,3,2)
        self.Conv3_1 = nn.Sequential(
            CBL(128,64,1,1),
            CBL(64,64,3,1)            
        )
        self.Conv3_2 = nn.Sequential(
            CBL(128,64,1,1),
            CBL(64,64,[7,1],1,(3,0)),
            CBL(64,64,[1,7],1,(0,3)),
            CBL(64,64,3,1)
        )
        self.Conv4_1 = CBL(128,128,3,2)
        self.maxpool4_2 = MaxPool2d(3,2)
        
    def forward(self,x):
        x = self.Conv1(x)
        x1 = self.maxpool2_1(x)
        x2 = self.Conv2_2(x)
        x = torch.cat((x1,x2),dim=1)
        x1 = self.Conv3_1(x)
        x2 = self.Conv3_2(x)
        x = torch.cat((x1,x2),dim=1)
        x1 = self.Conv4_1(x)
        x2 = self.maxpool4_2(x)
        x = torch.cat((x1,x2),dim=1)
        return x
    

class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.Conv1 = nn.Sequential(
            AvgPool2d(3,1,1),
            CBL(256,64,1)
        )
        self.Conv2 = CBL(256,64,1)
        self.Conv3 = nn.Sequential(
            CBL(256,64,1),
            CBL(64,64,3,1,1)
        )
        self.Conv4 = nn.Sequential(
            CBL(256,64,1),
            CBL(64,96,3,1,1),
            CBL(96,64,3,1,1)
        )
        
    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x4 = self.Conv4(x)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x

class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.Conv1 = nn.Sequential(
            AvgPool2d(3,1,1),
            CBL(512,128,1)
        )
        self.Conv2 = CBL(512,128,1)
        self.Conv3 = nn.Sequential(
            CBL(512,128,1),
            CBL(128,256,[1,7],1,(0,3)),
            CBL(256,128,[1,7],1,(0,3))
        )
        self.Conv4 = nn.Sequential(
            CBL(512,192,1),
            CBL(192,192,[1,7],1,(0,3)),
            CBL(192,224,[7,1],1,(3,0)),
            CBL(224,224,[1,7],1,(0,3)),
            CBL(224,128,[7,1],1,(3,0))
        )
        
    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x4 = self.Conv4(x)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()
        #1536通道换为1024
        #输出通道也变为1024
        self.Conv1 = nn.Sequential(
            AvgPool2d(3,1,1),
            CBL(512,128,1)
        )
        self.Conv2 = CBL(512,128,1)
        self.Conv3 = CBL(512,384,1)
        self.Conv3_1 = CBL(384,64,[1,3],1,(0,1))
        self.Conv3_2 = CBL(384,64,[3,1],1,(1,0))
        self.Conv4 = nn.Sequential(
            CBL(512,256,1),
            CBL(256,512,[1,3],1,(0,1)),
            CBL(512,256,[3,1],1,(1,0))
        )
        self.Conv4_1 = CBL(256,64,[3,1],1,(1,0))
        self.Conv4_2 = CBL(256,64,[1,3],1,(0,1))
    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = torch.cat((self.Conv3_1(self.Conv3(x)),self.Conv3_2(self.Conv3(x))),dim = 1)
        x4 = torch.cat((self.Conv4_1(self.Conv4(x)),self.Conv4_2(self.Conv4(x))),dim = 1)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x

class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.maxpool1 = MaxPool2d(3,2)
        self.Conv2 = CBL(256,128,3,2)
        self.Conv3 = nn.Sequential(
            CBL(256,128,1),
            CBL(128,256,3,1,1),
            CBL(256,128,3,2)
        )
        
    def forward(self,x):
        x1 = self.maxpool1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x = torch.cat((x1,x2,x3),dim=1)
        return x


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.maxpool1 = nn.Sequential(
            MaxPool2d(3,2),
            CBL(512,256,1)#加入一个1*1卷积，减少网络宽度。
        )
        self.Conv2 = nn.Sequential(
            CBL(512,192,1),
            CBL(192,128,3,2)
        )
        self.Conv3 = nn.Sequential(
            CBL(512,256,1),
            CBL(256,256,[1,7],1,(0,3)),
            CBL(256,320,[7,1],1,(3,0)),
            CBL(320,128,3,2)
        )
        
    def forward(self,x):
        x1 = self.maxpool1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x = torch.cat((x1,x2,x3),dim=1)
        return x

class Inception(nn.Module):
    def __init__(self,
                 in_channels
                ):
        super(Inception, self).__init__()
        self.stem = Stem(in_channels)
        self.Block_A = nn.Sequential(
            CBAM(256),
            Inception_A(),
            CBAM(256)
        )
        self.reduction_a = Reduction_A()
        self.Block_B = nn.Sequential(
            CBAM(512),
            Inception_B(),
            Inception_B(),
            CBAM(512)
        )
        self.reduction_b = Reduction_B()
        self.Block_C = nn.Sequential(
            CBAM(512),
            Inception_C(),
            CBAM(512)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(4, 4))
        self.classifier = nn.Sequential(
            Dropout(0.6),
            Linear(in_features=8192, out_features=4096, bias=True),
            ReLU(inplace=True),
            Dropout(0.6),
            Linear(in_features=4096, out_features=1024, bias=True),
            ReLU(inplace=True),
            Dropout(0.6),
            Linear(in_features=1024, out_features=3, bias=True)
        )
    
    def forward(self,x):
        x = self.stem(x)
        x = self.Block_A(x)
        x = self.reduction_a(x)
        x = self.Block_B(x)
        x = self.reduction_b(x)
        x = self.Block_C(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x

##########################################################

class Inception_FPN(nn.Module):
    def __init__(self,
                 in_channels
                ):
        super(Inception_FPN, self).__init__()
        self.stem = Stem(in_channels)
        self.Block_A = nn.Sequential(
            CBAM(384),
            Inception_A(),
            Inception_A(),
           # Inception_A(),
           # Inception_A(),
            CBAM(384)
        )
        self.reduction_a = Reduction_A()
        self.Block_B = nn.Sequential(
            Inception_B(),
            Inception_B(),
           # Inception_B(),
           # Inception_B(),
           # Inception_B(),
           # Inception_B(),
           # Inception_B(),
            CBAM(1024)
        )
        self.reduction_b = Reduction_B()
        self.Block_C = nn.Sequential(
            Inception_C(),
            Inception_C(),
           # Inception_C(),
            CBAM(1536)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(2, 2))
        self.linear1 = nn.Sequential(
          Dropout(0.5),
          Linear(1536,1024,bias = True)
          )
        self.linear2 = nn.Sequential(
          Dropout(0.5),
          Linear(4096,1024,bias = True)
        )
        self.linear3 = nn.Sequential(
          Dropout(0.5),
          Linear(6144,1024,bias = True)
          )
        self.classifier = nn.Sequential(
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(in_features=3072, out_features=3, bias=True)
        )
    
    def forward(self,x):
        x = self.stem(x)
        x1 = self.Block_A(x)
        x2 = self.reduction_a(x1)
        x2 = self.Block_B(x2)
        x3 = self.reduction_b(x2)
        x3 = self.Block_C(x3)
        x1 = self.avgpool(x1).view(x.shape[0],-1)
        x2 = self.avgpool(x2).view(x.shape[0],-1)
        x3 = self.avgpool(x3).view(x.shape[0],-1)
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)
        x = torch.cat((x1, x2, x3), 1)
       # x = x1 + x2 + x3
        x = self.classifier(x)
        return x
