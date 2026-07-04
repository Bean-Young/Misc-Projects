"""
CIFAR-10 data loading and preprocessing
Computer Vision Course Design 1
"""

import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
import os


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


def get_cifar10_data_loaders(data_dir, batch_size=128, num_workers=4):
    """
    Get CIFAR-10 data loaders with data augmentation

    Args:
        data_dir: Path to CIFAR-10 data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Create datasets
    train_dataset = CIFAR10Dataset(data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(data_dir, train=False, transform=test_transform)

    # Create data loaders
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
    # Test data loading
    tar_path = "/data/public/yyz/cv/cifar-10-python.tar.gz"
    extract_path = "/data/public/yyz/cv/data"

    # Extract data if not already extracted
    if not os.path.exists(os.path.join(extract_path, "cifar-10-batches-py")):
        extract_cifar10_data(tar_path, extract_path)

    data_dir = os.path.join(extract_path, "cifar-10-batches-py")

    # Test data loaders
    train_loader, test_loader = get_cifar10_data_loaders(data_dir, batch_size=32)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label range: {labels.min()} - {labels.max()}")
        break
