import cv2 as cv
import os
from matplotlib.pyplot import flag
import numpy as npy

def detect(filename, cascade_file,num,save_path):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv.CascadeClassifier(cascade_file)
    image = cv.imread(filename, cv.IMREAD_COLOR)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    i = int(0)
    flag = bool(0)
    for (x, y, w, h) in faces:
        i = i + 1
        flag = 1
        #cv.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 2)
        if w >= 90:
            try:
                out = image[y - 30 : y + h - 10, x - 10 : x + w + 10]
            except ValueError:
                continue
            else:
                try:
                    cv.imwrite(save_path + "\\" + str(num) + "_" + str(i) + ".png",out)
                except cv.error:
                    continue
    #if flag == 1:
    #    cv.rectangle(image, (x - 10, y - 20), (x + w + 10, y + h), (0, 0, 255), 2)
    #    cv.imshow("AnimeFaceDetect", image)
    #    cv.waitKey(0)
    #cv.imwrite("out_" + str(num) + ".png", image)

video_path = input("视频文件夹 : ")
#video_form = input("视频文件格式 : ")
save_path = input("图像保存地址 : ")
cascade_file = input("模型参数地址 : ")
file_list = os.listdir(video_path)
num = 0 
for name in file_list:
    if not os.path.isdir(name):
        video = cv.VideoCapture(video_path + "\\" + name)
        rval , frame = video.read()
        j = int(1)
        while rval:
            rval , frame = video.read()
            j = j + 1
            if j%96 == 0:
                num = num + 1
                cv.imwrite(save_path + "\\" + "tmp.png",frame)
                detect(save_path + "\\" + "tmp.png",cascade_file,num,save_path)