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

idea_size = 400
train_imgs_path = input("训练集文件夹地址 : ")
data = MyDataset.Data(train_imgs_path,idea_size,Device)
model_path = input("模型保存地址/生成器模型文件 : ")
G = MyModel.Generater(idea_size,Device).to(Device)
D = MyModel.Discriminater().to(Device)
if os.path.isdir(model_path):
    model_name = input("模型保存名称 : ")
    model_save_path = model_path
else:
    G = torch.load(model_path).to(Device)
    D = torch.load(input("Discriminater文件 : ")).to(Device)
    model_save_path = input("模型保存地址 : ")
    model_name = input("模型保存名称 : ")
testimgs_path = input("测试图片保存地址 : ")    

epoch = 2000#input("epoch : ")
Batch_size = 128#input("Batch_size : ")
loss_fn = nn.BCELoss().to(Device)
G_optimizer = torch.torch.optim.Adam(G.parameters(),lr=0.0002, betas=(0.5, 0.999))
D_optimizer = torch.torch.optim.Adam(D.parameters(),lr=0.0002, betas=(0.5, 0.999))
dataloader = DataLoader(data,batch_size = Batch_size,shuffle = True)

for i in range(epoch):
    G.train()
    D.train()
    print("Epoch : " + str(i+1))
    for T,img in enumerate(dataloader):
        if data.__len__()-(T*Batch_size)<=Batch_size:
            break
        D.zero_grad()
        label = torch.ones(Batch_size).to(Device)
        pred = D(img)
        pred = pred.reshape(-1)
        dloss_real = loss_fn(pred,label)#3
        dloss_real.backward()
        fake_label = torch.zeros(Batch_size).to(Device)
        #print(idea)
        idea = torch.randn([Batch_size,idea_size,1,1],device = Device)
        fake_imgs = G(idea)
        fake_pred = D(fake_imgs.detach())
        fake_pred = fake_pred.reshape(-1)
        dloss_fake = loss_fn(fake_pred,fake_label)#
        dloss_fake.backward()
        dloss = dloss_fake+dloss_real
        #D_optimizer.zero_grad()
        #dloss.backward()
        D_optimizer.step()
        if T%10 == 0:
            print("    Batch : " + str(T) + "    Discriminater loss : " + str(dloss.item()))
        if T > dataloader.__len__()/10:
            break
  
    for t in range(int(dataloader.__len__()/10)):
        G.zero_grad()
        g_label = torch.ones(Batch_size).to(Device)
        idea = torch.randn([Batch_size,idea_size,1,1],device = Device)
        fake_imgs = G(idea)
        g_pred = D(fake_imgs)
        g_pred = g_pred.reshape(-1)
        gloss = loss_fn(g_pred,g_label)
        #G_optimizer.zero_grad()
        gloss.backward()
        G_optimizer.step()
        G_loss = gloss.item()
        if t%10 == 0:
        #if t == 0:
            print("    Batch : " + str(t) + "    Generater loss : " + str(gloss.item())) #+ "   (times" + str(I) + ")")
        #if T > dataloader.__len__()/10:
        #    break

    if i % 20 == 0:
        torch.save(G, model_save_path + "\\" + model_name + "_G_" + str(i+1))
        torch.save(D, model_save_path + "\\" + model_name + "_D_" + str(i+1))
    #if i>50:
    #    os.remove(model_save_path + "\\" + model_name + "_G_" + str(i-50))
    #    os.remove(model_save_path + "\\" + model_name + "_D_" + str(i-50))
    if i%10 == 0:
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
    #D = MyModel.Discriminater()
    #G = MyModel.Generater(100)
    X = torch.randn((1,100,1,1))
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
        


