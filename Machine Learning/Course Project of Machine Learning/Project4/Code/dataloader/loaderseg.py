import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.transform = transform
        self.resize = transforms.Resize((512, 512), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx].strip()
        image_path = os.path.join(self.image_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # resize 图像与标签
        image = self.resize(image)
        label = self.resize(label)

        if self.transform:
            image = self.transform(image)

        # 转换 label 为 [H, W] 的 LongTensor，内容是类别编号（0~3）
        label = torch.from_numpy(np.array(label)).long()

        return image, label