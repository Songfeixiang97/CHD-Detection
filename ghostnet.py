#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import *

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


# In[19]:


class GhostModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s = 2
                 ):
        super(GhostModule,self).__init__()
        self.primary_conv = nn.Sequential(
            Conv2d(in_channels,out_channels//s,1,1,bias=False),
            BatchNorm2d(out_channels//2)
        )
        self.cheap_operation = nn.Sequential(
            Conv2d(out_channels//s,(out_channels//s)*(s-1),3,1,1,groups=out_channels//s,bias=False),
            BatchNorm2d((out_channels//s)*(s-1))
        )
    
    def forward(self,x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        x = torch.cat((x1,x2),dim=1)
        return x

class GhostBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 exp = 3,
                 attention = False
                 ):
        super(GhostBottleneck,self).__init__()
        if stride==1:
            self.conv1 = nn.Sequential(
                GhostModule(in_channels,in_channels*exp),
                ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                GhostModule(in_channels,in_channels*exp),
                ReLU(inplace=True),
                Conv2d(in_channels*exp,in_channels*exp,3,stride=stride,padding=1,groups=in_channels*exp, bias=False),
                BatchNorm2d(in_channels*exp),
                ReLU(inplace=True)
            )
        if attention==True:
            self.CBAM = CBAM(in_channels*exp)
        else:
            self.CBAM = nn.Sequential()
        self.conv2 = GhostModule(in_channels*exp,out_channels)
        if stride==1 and in_channels==out_channels:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                Conv2d(in_channels,in_channels,3,stride=stride,padding=1,groups=in_channels, bias=False),
                BatchNorm2d(in_channels),
                Conv2d(in_channels,out_channels,1,1,bias=False),
                BatchNorm2d(out_channels)
            )
    
    def forward(self,x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.CBAM(x)
        x = self.conv2(x)
        x = x + residual
        return x

class GhostNet(nn.Module):
    def __init__(self):
        super(GhostNet,self).__init__()
        self.conv = nn.Sequential(
            Conv2d(5,16,3,2,1,bias=False),
            BatchNorm2d(16),
            ReLU(inplace=True),
            
            GhostBottleneck(16,16,1,1,False),
            GhostBottleneck(16,24,3,2,False),
            
            GhostBottleneck(24,24,3,1,False),
            GhostBottleneck(24,40,3,2,True),
            
            GhostBottleneck(40,40,3,1,True),
            GhostBottleneck(40,80,6,2,False),
            
            GhostBottleneck(80,80,3,1,False),
            GhostBottleneck(80,120,4,1,True),
            GhostBottleneck(120,160,4,2,True),
            
            GhostBottleneck(160,160,4,1,False),
            GhostBottleneck(160,160,4,1,True),
            
            Conv2d(160,960,1,1,bias=False),
            BatchNorm2d(960),
            ReLU(inplace=True),
            AdaptiveAvgPool2d((1, 1)),
            Conv2d(960,1280,1,1,bias=True),
            ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            Dropout(0.5),
            Linear(1280,3)
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x