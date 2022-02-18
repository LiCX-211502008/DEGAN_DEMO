from multiprocessing.context import set_spawning_popen
from pickletools import uint8
from random import random
from re import I
from turtle import forward
import numpy as npy
import cv2 as cv
import struct
import time
import torch
from torch import device, nn, tensor
from torch.utils.data import DataLoader
from torchaudio import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.onnx as onnx
import torchvision.models as models


class Generater(nn.Module):
    def __init__(self,input_size,Device) -> None:
        super().__init__()
        self.Device = Device
        self.size = input_size
        self.ConvT1 = nn.ConvTranspose2d(in_channels = self.size,
                                        out_channels = 512,
                                        kernel_size = 8,
                                        stride = 1,
                                        padding = 0,
                                        bias = False)
        nn.init.normal_(self.ConvT1.weight.data, 0, 0.02)
        self.BN1 = nn.BatchNorm2d(num_features = 512)#8*8
        nn.init.normal_(self.BN1.weight.data, 0, 0.02)
        nn.init.constant_(self.BN1.bias.data, 0)
        self.ReLU1 = nn.ReLU()
        self.ConvT2 = nn.ConvTranspose2d(in_channels = 512,
                                        out_channels = 256,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = False)
        nn.init.normal_(self.ConvT2.weight.data, 0, 0.02)
        self.BN2 = nn.BatchNorm2d(num_features = 256)#16*16
        nn.init.normal_(self.BN2.weight.data, 0, 0.02)
        nn.init.constant_(self.BN2.bias.data, 0)
        self.ReLU2 = nn.ReLU()
        self.ConvT3 = nn.ConvTranspose2d(in_channels = 256,
                                        out_channels = 128,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = False
                                        )
        nn.init.normal_(self.ConvT3.weight.data, 0, 0.02)
        self.BN3 = nn.BatchNorm2d(num_features = 128)#32*32
        nn.init.normal_(self.BN3.weight.data, 0, 0.02)
        nn.init.constant_(self.BN3.bias.data, 0)
        self.ReLU3 = nn.ReLU()
        self.ConvT4 = nn.ConvTranspose2d(in_channels = 128,
                                        out_channels = 3,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = False
                                        )
        nn.init.normal_(self.ConvT4.weight.data, 0, 0.02)
        self.BN4 = nn.BatchNorm2d(num_features = 64)#64*64
        nn.init.normal_(self.BN4.weight.data, 0, 0.02)
        nn.init.constant_(self.BN4.bias.data, 0)
        self.ReLU4 = nn.ReLU()
        self.ConvT5 = nn.ConvTranspose2d(in_channels = 64,
                                        out_channels = 3,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = False
                                        )
        nn.init.normal_(self.ConvT5.weight.data, 0, 0.02)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        out = self.ConvT1(x).to(self.Device)
        out = self.BN1(out)
        out = self.ReLU1(out)
        out = self.ConvT2(out)
        out = self.BN2(out)
        out = self.ReLU2(out)
        out = self.ConvT3(out)
        out = self.BN3(out)
        out = self.ReLU3(out)
        out = self.ConvT4(out)
        #out = self.BN4(out)
        #out = self.ReLU4(out)
        #out = self.ConvT5(out)
        out = self.tanh(out)
        return out

class Discriminater(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminater = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(num_features = 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(num_features = 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(num_features = 512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 512,out_channels = 1,kernel_size = 4,stride = 2,padding = 0,bias = False),
            #nn.LeakyReLU(0.2),
            #nn.Conv2d(in_channels = 1024,out_channels = 1,kernel_size = 3,stride = 2,padding = 0,bias = False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.discriminater(x)
        out = out.view(x.size(0),-1)
        return out


'''
X = torch.randn([1,3,64,64])
D = Discriminater()
F = D(X)
print(F.size())
'''
'''
D = Discriminater()
X = torch.randn((1,100,1,1))
G = Generater(100)
T = G(X)
img = T.detach().numpy()
img *= 100
show = npy.zeros((64,64,3))
show[:,:,0] = img[0,0]
show[:,:,1] = img[0,1]
show[:,:,2] = img[0,2]
show = cv.resize(show,(500,500))
cv.imshow("test",show)
cv.waitKey(0)
print(npy.shape(img))
F =D(T)
print(F)
'''