import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data_fashion_mnist(batch_size, resize=None):
    """加载Fashion-MNIST数据集"""
    # 定义数据转换
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    
    # 指定数据路径
    data_path = "./NNDL-Class/Project1/Data/"
    
    # 加载训练集和测试集
    mnist_train = datasets.FashionMNIST(
        root=data_path, train=True, transform=transform, download=False)
    mnist_test = datasets.FashionMNIST(
        root=data_path, train=False, transform=transform, download=False)
    
    # 创建数据加载器
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    return train_iter, test_iter

# 训练函数
def train_model(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 记录训练和测试精度、损失
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        # 训练
        net.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.size(0)
        
        train_loss = train_loss_sum / len(train_iter)
        train_acc = train_acc_sum / n
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # 测试
        net.eval()
        test_loss_sum, test_acc_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                test_loss_sum += l.item()
                test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.size(0)
        
        test_loss = test_loss_sum / len(test_iter)
        test_acc = test_acc_sum / n
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        
        print(f'epoch {epoch+1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, '
              f'test loss {test_loss:.4f}, test acc {test_acc:.4f}')
    
    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


# 实现Residual块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, 
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 
                              kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 测试残差块
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print("输入输出形状相同的残差块输出形状:", Y.shape)

# 输入输出通道数不同的残差块
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print("通道数改变的残差块输出形状:", blk(X).shape)

# 构建ResNet-18
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 构建完整的ResNet-18模型
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(512, 10)
)

# 检查各层输出形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 在Fashion-MNIST上训练ResNet
lr, num_epochs, batch_size = 0.05, 10, 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互式模式

train_loss, train_acc, test_loss, test_acc = train_model(
    net, train_iter, test_iter, num_epochs, lr, device)

# 训练结束后再显示最终结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('ResNet Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_acc, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('ResNet Accuracy Curves')

plt.tight_layout()
plt.savefig('./NNDL-Class/Project3/Result/resnet_performance.png')

