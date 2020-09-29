#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid, AdaptiveAvgPool1d, AdaptiveMaxPool1d, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, BatchNorm1d, LeakyReLU, Dropout2d, UpsamplingBilinear2d, AdaptiveMaxPool2d

class DSC(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 1
                ):
        super(DSC, self).__init__()
        self.conv1 = Conv2d(in_channels, 
                            in_channels, 
                            kernel_size, 
                            stride, 
                            padding = padding, 
                            groups = in_channels
                           )
        self.conv2 = Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
            DSC(5,64,3,2,1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            DSC(64,192,3,2,1),
            BatchNorm2d(192),
            ReLU(inplace=True),
            DSC(192,384,3,1,1),
            BatchNorm2d(384),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            DSC(384,512,3,1,1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            DSC(512,512,3,1,1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            DSC(512,1024,3,1,1),
            BatchNorm2d(1024),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            DSC(1024,1024,3,1,1),
            BatchNorm2d(1024),
            ReLU(inplace=True),
            DSC(1024,512,3,1,1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            DSC(512,256,3,1,1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            CBAM(256)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(
            Dropout(p=0.5),
            Linear(in_features = 9216, out_features=2048, bias=True),
            ReLU(inplace = True),
            Dropout(p=0.5),
            Linear(in_features = 2048, out_features=1024, bias=True),
            ReLU(inplace = True),
            Dropout(p=0.5),
            Linear(in_features=1024, out_features=3, bias=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x



class CBL(nn.Module):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = [3,3],
                 stride = 1,
                 padding = 1
                 ):
        super(CBL, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.BatchNorm2d = BatchNorm2d(out_channels)
        self.LeakyReLU = LeakyReLU(0.01, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BatchNorm2d(x)
        x = self.LeakyReLU(x)
        return x
    
class CBLR_DSC(nn.Module):
    def __init__(self,
                 in_channels = None,
                 kernel_size = [3,3],
                 stride = 1,
                 padding = 1
                 ):
        super(CBLR_DSC, self).__init__()
        self.cbl = CBL(in_channels, in_channels//2, [1, 1], stride, padding = 0)
        self.conv = DSC(in_channels//2, in_channels, kernel_size, stride, padding)
        self.BatchNorm2d = BatchNorm2d(in_channels)
        self.LeakyReLU = LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        y = self.cbl(x)
        y = self.conv(y)
        y = y.add(x)
        y = self.BatchNorm2d(y)
        y = self.LeakyReLU(y)
        return y
    
class CBL(nn.Module):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = [3,3],
                 stride = 1,
                 padding = 1
                 ):
        super(CBL, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.BatchNorm2d = BatchNorm2d(out_channels)
        self.LeakyReLU = LeakyReLU(0.01, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BatchNorm2d(x)
        x = self.LeakyReLU(x)
        return x

class CBLR(nn.Module):
    def __init__(self,
                 in_channels = None,
                 kernel_size = [3,3],
                 stride = 1,
                 padding = 1
                 ):
        super(CBLR, self).__init__()
        self.cbl = CBL(in_channels, in_channels//2, [1, 1], stride, padding = 0)
        self.conv = Conv2d(in_channels//2, in_channels, kernel_size, stride, padding)
        self.BatchNorm2d = BatchNorm2d(in_channels)
        self.LeakyReLU = LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        y = self.cbl(x)
        y = self.conv(y)
        y = y.add(x)
        y = self.BatchNorm2d(y)
        y = self.LeakyReLU(y)
        return y

class CBL_set(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels
                ):
        super(CBL_set, self).__init__()
        self.conv1 = CBL(in_channels, out_channels, [1,1], 1, 0)
        self.conv2 = CBL(out_channels, out_channels, [3,3])
        self.conv3 = CBL(out_channels, 2*out_channels, [1,1], 1, 0)
        self.conv4 = CBL(2*out_channels, 2*out_channels, [3,3])
        self.conv5 = CBL(2*out_channels, out_channels, [1,1], 1, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Darknet53(nn.Module):
    def __init__(self,
                 in_channels
                 ):
        super(Darknet53, self).__init__()
        self.cbl1 = CBL(in_channels, 64)
        self.conv1 = Conv2d(64, 128, [3,3], 2, 1)
        self.cblr1 = nn.Sequential(
            CBLR(128),
            CBAM(128)
        )
        self.conv2 = Conv2d(128, 192, [3,3], 2, 1)
        self.cblr2 = nn.Sequential(
            CBLR(192),
            CBLR(192),
            CBAM(192)
        )
        self.conv3 = Conv2d(192, 256, [3,3], 2, 1)
        self.cblr3 = nn.Sequential(
            CBLR(256), 
            CBLR(256), 
            CBAM(256)
        )
        self.conv4 = Conv2d(256, 512, [3,3], 2, 1)
        self.cblr4 = nn.Sequential(
            CBLR(512), 
            CBLR(512),
            CBAM(512)
        )
        self.conv5 = Conv2d(512, 1024, [3,3], 2, 1)
        self.cblr5 = nn.Sequential(
            CBLR(1024), 
            CBLR(1024), 
            CBAM(1024)
        )
        self.CBL_set1 = CBL_set(1024, 1024)
        self.cbl2 = CBL(1024, 1024, [3,3])
        self.conv6 = nn.Sequential(
            Conv2d(1024, 64, 1, 1),
            CBAM(64)
        )
        self.conv7 = Conv2d(1024, 256, 1, 1)
        self.CBL_set2 = CBL_set(768, 256)
        self.cbl3 = CBL(256, 256, [3,3])
        self.conv8 = nn.Sequential(
            Conv2d(256, 64, 1, 1),
            CBAM(64)
        )
        self.conv9 = Conv2d(256, 128, 1, 1)
        self.CBL_set3 = CBL_set(384, 128)
        self.cbl4 = CBL(128, 128, [3,3])
        self.conv10 = nn.Sequential(
            Conv2d(128, 64, 1, 1),
            CBAM(64)
        )
        
    def forward(self, x):
        x = self.cbl1(x)
        x = self.conv1(x)
        x = self.cblr1(x)
        x = self.conv2(x)
        x = self.cblr2(x)
        x = self.conv3(x)
        out1 = self.cblr3(x)
        x = self.conv4(out1)
        out2 = self.cblr4(x)
        x = self.conv5(out2)
        out3 = self.cblr5(x)
        out3 = self.CBL_set1(out3)
        y1 = self.cbl2(out3)
        y1 = self.conv6(y1)
        out3 = self.conv7(out3)
        out3 = UpsamplingBilinear2d(scale_factor=2)(out3)
        out3 = torch.cat((out3,out2),1)
        out3 = self.CBL_set2(out3)
        y2 = self.cbl3(out3)
        y2 = self.conv8(out3)
        out3 = self.conv9(out3)
        out3 = UpsamplingBilinear2d(scale_factor=2)(out3)
        out3 = torch.cat((out3,out1),1)
        out3 = self.CBL_set3(out3)
        y3 = self.cbl4(out3)
        y3 = self.conv10(y3)
        return y1,y2,y3
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.darknet53 = Darknet53(5)
        self.avgpool = AdaptiveAvgPool2d(output_size=(6,6))
        self.linear1 = nn.Sequential(
            Dropout(0.5),
            Linear(2304,1024,bias = True)
        )
        self.linear2 = nn.Sequential(
            Dropout(0.5),
            Linear(2304,1024,bias = True)
        )
        self.linear3 = nn.Sequential(
            Dropout(0.5),
            Linear(2304,1024,bias = True)
        )
        self.classifier = nn.Sequential(
            LeakyReLU(0.01, inplace=True),
            Dropout(p=0.5),
            Linear(in_features=3072, out_features=3, bias=True),
        )
        
    def forward(self, x):
        y1, y2, y3 = self.darknet53(x)
        y1 = self.avgpool(y1).view(x.shape[0],-1)
        y2 = self.avgpool(y2).view(x.shape[0],-1)
        y3 = self.avgpool(y3).view(x.shape[0],-1)
        y1 = self.linear1(y1)
        y2 = self.linear2(y2)
        y3 = self.linear3(y3)
        y = torch.cat((y1, y2, y3), 1)
        y = self.classifier(y)
        return y
