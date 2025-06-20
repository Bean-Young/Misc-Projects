import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18ForCIFAR100(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18ForCIFAR100, self).__init__()
        
        # 加载预训练的ResNet18模型
        model = models.resnet18(pretrained=pretrained)
        
        # 修改第一个卷积层以适应CIFAR100的32x32输入
        # 原始ResNet设计用于224x224的ImageNet图像
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # 移除最大池化层，因为CIFAR100图像较小
        
        # 修改最后的全连接层以输出100个类别（CIFAR100的类别数）
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100)
        
        self.model = model
        
    def forward(self, x):
        return self.model(x)
