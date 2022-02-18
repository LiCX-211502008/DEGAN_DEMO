from pyexpat import model
import WGAN_Model
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


def GP(net,device,real,fake):
    alpha = torch.randn(real.size(0),1,requires_grad = True)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device)

    interpolates = alpha * real + (1 - alpha) * fake
    D_interpolates = net(interpolates)

    grad = torch.autograd.grad(outputs = D_interpolates,inputs = interpolates,grad_outputs = torch.ones(D_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((grad.norm(2,dim = 1) - 1) ** 2).mean()
    return gp



Device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(Device))

idea_size = 100
train_imgs_path = input("训练集文件夹地址 : ")
data = MyDataset.Data(train_imgs_path,idea_size,Device)
model_path = input("模型保存地址/生成器模型文件 : ")
G = WGAN_Model.Generater(idea_size,Device).to(Device)
D = WGAN_Model.Discriminater().to(Device)
if os.path.isdir(model_path):
    model_name = input("模型保存名称 : ")
    model_save_path = model_path
else:
    G = torch.load(model_path).to(Device)
    D = torch.load(input("Discriminater文件 : ")).to(Device)
    model_save_path = input("模型保存地址 : ")
    model_name = input("模型保存名称 : ")
testimgs_path = input("测试图片保存地址 : ")    

epoch = 1000000#input("训练次数 : ")
Batch_size = 64#input("Batch_size : ")
G_optimizer = torch.torch.optim.Adam(G.parameters(),lr = 0.0001,betas = (0,0.9))
D_optimizer = torch.torch.optim.Adam(D.parameters(),lr = 0.0001,betas = (0,0.9))
dataloader = DataLoader(data,batch_size = Batch_size,shuffle = True)
c = float(0.01)
n_critic = int(5)
for i in range(epoch):
    G.train()
    D.train()
    print("Epoch : " + str(i+1))
    times = int(0)
    for T,img in enumerate(dataloader):
        if data.__len__() - (T*Batch_size) <= Batch_size:
            break
        times = times + 1
        D.zero_grad()
        pred = D(img)
        dvalue_real = torch.mean(pred)
        idea = torch.randn([Batch_size,idea_size,1,1],device = Device)
        fake_imgs = G(idea)
        fake_imgs = fake_imgs.detach()
        fake_pred = D(fake_imgs)
        dvalue_fake = torch.mean(fake_pred)
        gp = GP(D,Device,img,fake_imgs)
        dvalue = -(dvalue_real-dvalue_fake) + gp*10
        dvalue.backward()
        D_optimizer.step()
        print("    Batch : " + str(T) + "    Discriminater value : " + str(-dvalue.item()))
  
        if times % n_critic == 0:
            for _ in range(2):
                G.zero_grad()
                idea = torch.randn([Batch_size,idea_size,1,1],device = Device)
                fake_imgs = G(idea)
                g_pred = D(fake_imgs)
                gvalue = -torch.mean(g_pred)
                gvalue.backward()
                G_optimizer.step()
                print("    Batch : " + str(T) + "    Generater value : " + str(-gvalue.item()))

    torch.save(G, model_save_path + "\\" + model_name + "_G_" + str(i+1))
    torch.save(D, model_save_path + "\\" + model_name + "_D_" + str(i+1))
    if i>50:
        os.remove(model_save_path + "\\" + model_name + "_G_" + str(i-50))
        os.remove(model_save_path + "\\" + model_name + "_D_" + str(i-50))
    if i % 50 == 0:
        G.eval()
        D.eval()
        test = torch.randn([1,idea_size,1,1]).to(Device)
        T = G(test)
        T = T.to("cpu")
        img = T.detach().numpy()
        show = npy.zeros((64,64,3))
        show[:,:,0] = img[0,0]
        show[:,:,1] = img[0,1]
        show[:,:,2] = img[0,2]
        show = show * 255
        if not os.path.exists(testimgs_path):
            os.mkdir(testimgs_path)
        print("    " + testimgs_path + "\\" + model_name +"_Epoch_" + str(i) + ".png")
        cv.imwrite(testimgs_path + "\\" + model_name +"_Epoch_" + str(i) + ".png",show)
    
while 1:
    D = WGAN_Model.Discriminater()
    X = torch.randn((1,100,1,1))
    G = WGAN_Model.Generater(100)
    T = G(X)
    img = T.detach().numpy()
    img *= 255
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
        


