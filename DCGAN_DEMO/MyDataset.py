from cmath import tanh
from multiprocessing.context import set_spawning_popen
from pickletools import uint8
from random import random
from turtle import forward
import numpy as npy
import cv2 as cv
import struct
import time
import torch
from torch import device, nn
from torch.utils.data import DataLoader
from torchaudio import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.onnx as onnx
import torchvision.models as models
import MyModel
import os

class Data(Dataset,nn.Module):
    def __init__(self,path,idea_size,Device) -> None:
        super().__init__()
        self.Device = Device
        self.path = path
        self.file_list = []
        for i in os.listdir(path):
            if not os.path.isdir(i):
                self.file_list.append(i)
        torch.manual_seed(len(self.file_list))
        self.idea_size = idea_size

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img = cv.imread(self.path + "\\" + self.file_list[index])
        img = cv.resize(img,[64,64])
        img = torch.from_numpy(img).to(self.Device)
        Y = torch.zeros([3,64,64]).to(self.Device)
        Y[0] = img[:,:,0]
        Y[1] = img[:,:,1]
        Y[2] = img[:,:,2]
        Y = Y/255
        return  Y
