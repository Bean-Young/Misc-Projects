"""
增强的数据加载器 - 更多数据增强技术
Computer Vision Course Design 1
"""

import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
import os
from PIL import Image
import random


class CIFAR10Dataset:
    """CIFAR-10 dataset loader"""

    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        self.data = []
        self.targets = []

        if train:
            self._load_train_data()
        else:
            self._load_test_data()

    def _load_train_data(self):
        """Load training data from all batches"""
        for i in range(1, 6):
            file_path = os.path.join(self.data_dir, f'data_batch_{i}')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])
                self.targets.extend(batch[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format

    def _load_test_data(self):
        """Load test data"""
        file_path = os.path.join(self.data_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data'].reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
            self.targets = batch[b'labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Cutout:
    """Cutout数据增强"""
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MixUp:
    """MixUp数据增强"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        data, targets = batch
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_data, targets_a, targets_b, lam


def get_enhanced_cifar10_data_loaders(data_dir, batch_size=128, num_workers=4, use_mixup=False):
    """
    获取增强的CIFAR-10数据加载器

    Args:
        data_dir: CIFAR-10数据目录
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        use_mixup: 是否使用MixUp增强

    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """

    # 增强的训练数据变换
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(length=16)  # Cutout增强
    ])

    # 测试数据变换（无增强）
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 创建数据集
    train_dataset = CIFAR10Dataset(data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(data_dir, train=False, transform=test_transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def extract_cifar10_data(tar_path, extract_path):
    """
    Extract CIFAR-10 tar.gz file

    Args:
        tar_path: Path to cifar-10-python.tar.gz
        extract_path: Path to extract the data
    """
    import tarfile

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_path)

    print(f"CIFAR-10 data extracted to {extract_path}")


if __name__ == "__main__":
    # Test enhanced data loading
    tar_path = "/data/public/yyz/cv/cifar-10-python.tar.gz"
    extract_path = "/data/public/yyz/cv/data"

    # Extract data if not already extracted
    if not os.path.exists(os.path.join(extract_path, "cifar-10-batches-py")):
        extract_cifar10_data(tar_path, extract_path)

    data_dir = os.path.join(extract_path, "cifar-10-batches-py")

    # Test enhanced data loaders
    train_loader, test_loader = get_enhanced_cifar10_data_loaders(data_dir, batch_size=32)

    print(f"Enhanced training samples: {len(train_loader.dataset)}")
    print(f"Enhanced test samples: {len(test_loader.dataset)}")

    # Test a batch
    for images, labels in train_loader:
        print(f"Enhanced batch shape: {images.shape}")
        print(f"Enhanced labels shape: {labels.shape}")
        print(f"Enhanced label range: {labels.min()} - {labels.max()}")
        print(f"Enhanced image range: {images.min():.3f} - {images.max():.3f}")
        break
