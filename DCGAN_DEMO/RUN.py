from pyexpat import model
import MyModel
import MyDataset
import os
from multiprocessing.context import set_spawning_popen
from pickletools import uint8
from random import random, shuffle
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

Device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(Device))
G = torch.load(input("生成器模型 : ")).to(Device)

while 1:
    X = torch.randn((1,400,1,1)).to(Device)
    T = G(X).to("cpu")
    img = T.detach().numpy()
    #img *= 255
    show = npy.zeros((64,64,3))
    show[:,:,0] = img[0,0]
    show[:,:,1] = img[0,1]
    show[:,:,2] = img[0,2]
    show = cv.resize(show,(500,500))
    cv.imshow("test",show)
    cv.waitKey(0)
    print(npy.shape(img))