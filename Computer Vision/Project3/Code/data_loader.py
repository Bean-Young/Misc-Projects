import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=128, num_workers=2):

    # 数据预处理和增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./Data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root='./Data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    # CIFAR10数据集的类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes
