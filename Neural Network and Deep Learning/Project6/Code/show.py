import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from d2l import torch as d2l
import torchvision

# 创建保存目录
save_dir = './NNDL-Class/Project6/Result/seg'
os.makedirs(save_dir, exist_ok=True)

PALETTE = [
    [0, 0, 0],        # 背景
    [128, 0, 0],      # 飞机
    [0, 128, 0],      # 汽车
    [128, 128, 0],    # 汽船
    [0, 0, 128],      # 鸟
    [128, 0, 128],    # 猫
    [0, 128, 128],    # 奶牛
    [128, 128, 128],  # 狗
    [64, 0, 0],       # 马
    [192, 0, 0],      # 羊
    [64, 128, 0],     # 牛
    [192, 128, 0],    # 飞行器
    [64, 0, 128],     # 车辆
    [192, 0, 128],    # 卡车
    [64, 128, 128],   # 动物
    [192, 128, 128],  # 建筑
    [0, 64, 0],       # 运输工具
    [128, 64, 0],     # 水果
    [0, 64, 128],     # 工业
    [128, 64, 128]    # 其他
]

def read_voc_images(voc_dir, is_train=True):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    features, labels = [], []
    for i, fname in enumerate(images):
        img_path = os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')
        label_path = os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png')
        
        img = Image.open(img_path).convert('RGB')  # 修正：使用 .open() 和 .convert('RGB')
        label = Image.open(label_path).convert('RGB')  # 修正：使用 .open() 和 .convert('RGB')
        
        features.append(torchvision.transforms.ToTensor()(img))
        labels.append(torchvision.transforms.ToTensor()(label))
        
    return features, labels

def voc_rand_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def show_first_five_classes(voc_dir):
    # 读取数据
    train_features, train_labels = read_voc_images(voc_dir, True)

    # 设置图像显示
    fig, axs = plt.subplots(5, 2, figsize=(15, 15))

    # 显示前五个类别的图像和标签
    for i in range(5):
        img = train_features[i]
        label = train_labels[i]
        
        # 显示图像
        axs[i, 0].imshow(img.permute(1, 2, 0))  # 转换为(高度, 宽度, 通道)
        axs[i, 0].set_title(f'Class {i+1} Image')
        axs[i, 0].axis('off')

        # 显示标签
        axs[i, 1].imshow(label.permute(1, 2, 0))  # 转换为(高度, 宽度, 通道)
        axs[i, 1].set_title(f'Class {i+1} Label')
        axs[i, 1].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'first_five_classes_with_labels.png'))
    plt.show()

def show_random_crop(voc_dir):
    # 读取数据
    train_features, train_labels = read_voc_images(voc_dir, True)

    # 选择四个不同的图像进行裁剪展示
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        feature = train_features[i]
        label = train_labels[i]

        # 随机裁剪
        cropped_feature, cropped_label = voc_rand_crop(feature, label, 200, 300)

        # 显示原始图像与裁剪后的图像
        axs[0, i].imshow(feature.permute(1, 2, 0))  # 原始图像
        axs[0, i].set_title(f'Original Image {i+1}')
        axs[0, i].axis('off')

        axs[1, i].imshow(cropped_feature.permute(1, 2, 0))  # 裁剪后的图像
        axs[1, i].set_title(f'Cropped Image {i+1}')
        axs[1, i].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'random_crop_examples.png'))
    plt.show()

# 调用函数显示前五个类别的示例及标签
voc_dir = './NNDL-Class/Project6/Data/VOCdevkit/VOC2012'
show_first_five_classes(voc_dir)

# 调用函数显示随机裁剪的示例
show_random_crop(voc_dir)
