import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 数据预处理
transformos = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
#train_dataset = ImageFolder(root='dataset/train', transform=transform)
#test_dataset = ImageFolder(root='dataset/test', transform=transform)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)