from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata
from Vgg16 import VGG16



import sys
import imp
imp.reload(sys)


dataset_dir = 'E:/Python/Project/Cat vs Dog/cat_dog/train/'                    # 数据集路径
model_file = 'E:/Python/Project/Cat vs Dog/DogsVsCats/DogsVsCats-master/model(VGG16)/model.pth'                # 模型保存路径
N = 20

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# .to(device)
# # old version
# def test():
#
#     model = Net()                                       # 实例化一个网络
#     model.cuda()                                        # 送入GPU，利用GPU计算
#     model = nn.DataParallel(model)       +
#     model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
#     model.eval()                                        # 设定为评估模式，即计算过程中不要dropout
#
#     datafile = DVCD('test', dataset_dir)                # 实例化一个数据集
#     print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
#
#     index = np.random.randint(0, datafile.data_size, 1)[0]      # 获取一个随机数，即随机从数据集中获取一个测试图片
#     img = datafile.__getitem__(index)                           # 获取一个图像
#     img = img.unsqueeze(0)                                      # 因为网络的输入是一个4维Tensor，3维数据，1维样本大小，所以直接获取的图像数据需要增加1个维度
#     img = Variable(img).cuda()                                  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
#     print(img)
#     out = model(img)                                            # 网路前向计算，输出图片属于猫或狗的概率，第一列维猫的概率，第二列为狗的概率
#     out = F.softmax(out, dim=1)                                        # 采用SoftMax方法将输出的2个输出值调整至[0.0, 1.0],两者和为1
#     print(out)                      # 输出该图像属于猫或狗的概率
#     if out[0, 0] > out[0, 1]:                   # 猫的概率大于狗
#         print('the image is a cat')
#     else:                                       # 猫的概率小于狗
#         print('the image is a dog')
#
#     img = Image.open(datafile.list_img[index])      # 打开测试的图片
#     plt.figure('image')                             # 利用matplotlib库显示图片
#     plt.imshow(img)
#     plt.show()


# new version
def test():

    # setting model
    model =VGG16()                                       # 实例化一个网络
    model.to(device)                                       # 送入GPU，利用GPU计算
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    list_label=[]
    ans=0
    for file in files:
        img = Image.open(dataset_dir + file)            # 打开图像
        img_data = getdata.dataTransform(img).to(device)           # 转换成torch tensor数据   ++
        name = file.split(sep='_')
        if name[0] == 'cat':
            list_label.append(0)
        else:
            list_label.append(1)
        imgs.append(img)                                # 图像list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list合成一个4D tensor

    # calculation
    out = model(imgs_data)                              # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)                         # 输出概率化
    out = out.data.cpu().numpy()                        # 转成numpy数据

    # pring results         显示结果
    for idx in range(N):
        if out[idx, 0] > out[idx, 1]:
            if list_label[idx]==0 :
                ans+=1
        else:
            if list_label[idx]==1 :
                ans+=1
    print("Test Accuracy:",(ans/N)*100,"%")
if __name__ == '__main__':
    test()


