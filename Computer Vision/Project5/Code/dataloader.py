import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode
# PASCAL VOC 数据集的颜色映射（RGB格式）
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

# Pascal VOC数据集加载器 - 修复后的版本
class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
        
        # 获取文件列表
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        self.file_names = [line.strip() for line in open(split_file, 'r')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.images_dir, img_name + '.jpg')
        mask_path = os.path.join(self.masks_dir, img_name + '.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # 应用相同的空间变换到图像和掩码
        if self.transform:
            # 对图像应用完整的transform
            image = self.transform(image)
            
            # 对掩码只应用空间变换（Resize），使用新的插值模式
            mask = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)(mask)
            mask = self.rgb_to_index(np.array(mask))
            mask = torch.from_numpy(mask).long()
        else:
            # 如果没有transform，只进行基本处理
            mask = self.rgb_to_index(np.array(mask))
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def rgb_to_index(self, mask_rgb):
        """将RGB掩码转换为类别索引图"""
        h, w, _ = mask_rgb.shape
        mask_index = np.zeros((h, w), dtype=np.uint8)
        
        # 创建颜色到索引的映射
        color_to_index = {}
        for idx, color in enumerate(PALETTE):
            color_tuple = tuple(color)
            color_to_index[color_tuple] = idx
        
        # 转换每个像素
        for i in range(h):
            for j in range(w):
                pixel = tuple(mask_rgb[i, j])
                mask_index[i, j] = color_to_index.get(pixel, 0)  # 未知颜色默认为背景
        
        return mask_index

def get_dataloader(batch_size):
    # 数据集路径 - 根据您的要求修改
    root_dir = './Data/VOCdevkit/VOC2012'
    
    # 图像预处理 - 只对图像应用这些转换
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集 - 使用相同的transform对象
    train_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        split='train',
        transform=image_transform
    )
    
    val_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        split='val',
        transform=image_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader