import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

plt.ioff()  # 关闭交互式模式以防止图表显示

# 基础LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sigmoid1 = nn.Sigmoid()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.sigmoid3 = nn.Sigmoid()
        self.fc2 = nn.Linear(120, 84)
        self.sigmoid4 = nn.Sigmoid()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.avgpool1(self.sigmoid1(self.conv1(x)))
        x = self.avgpool2(self.sigmoid2(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid3(self.fc1(x))
        x = self.sigmoid4(self.fc2(x))
        x = self.fc3(x)
        return x

# 带BatchNorm的LeNet
class LeNetBN(nn.Module):
    def __init__(self):
        super(LeNetBN, self).__init__()
        # 第一个带BN的卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.sigmoid1 = nn.Sigmoid()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二个带BN的卷积层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 带BN的全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.sigmoid3 = nn.Sigmoid()
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.sigmoid4 = nn.Sigmoid()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.avgpool1(self.sigmoid1(self.bn1(self.conv1(x))))
        x = self.avgpool2(self.sigmoid2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.sigmoid3(self.bn3(self.fc1(x)))
        x = self.sigmoid4(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

# 练习题2：改进的LeNet模型
class ImprovedLeNet(nn.Module):
    def __init__(self):
        super(ImprovedLeNet, self).__init__()
        # 1. 修改卷积窗口大小
        # 2. 增加输出通道数
        # 3. 使用ReLU激活函数
        # 4. 增加一个卷积层
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积层（新增）
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout1(self.relu4(self.bn4(self.fc1(x))))
        x = self.dropout2(self.relu5(self.bn5(self.fc2(x))))
        x = self.fc3(x)
        return x

# 训练函数
def train_model(net, train_iter, test_iter, num_epochs, lr, device, scheduler=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('train on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
        
        # 如果提供了学习率调度器，则更新学习率
        if scheduler:
            scheduler.step()
        
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, '
      f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history

# 加载数据集
transform = transforms.Compose([transforms.ToTensor()])
data_root = './NNDL-Class/Project1/Data'
train_dataset = FashionMNIST(root=data_root, train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root=data_root, train=False, transform=transform, download=True)

batch_size = 256
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练原始LeNet模型
lr, num_epochs = 0.9, 10
net = LeNet()
train_loss, train_acc, test_loss, test_acc = train_model(
    net, train_iter, test_iter, num_epochs, lr, device)

# 训练带BatchNorm的LeNet
net_bn = LeNetBN()
train_loss_bn, train_acc_bn, test_loss_bn, test_acc_bn = train_model(
    net_bn, train_iter, test_iter, num_epochs, lr, device)

# 训练改进的LeNet（练习题2）
lr_improved = 0.05
improved_net = ImprovedLeNet()
scheduler = torch.optim.lr_scheduler.StepLR(
    torch.optim.SGD(improved_net.parameters(), lr=lr_improved, momentum=0.9, weight_decay=5e-4), 
    step_size=3, gamma=0.5)
train_loss_improved, train_acc_improved, test_loss_improved, test_acc_improved = train_model(
    improved_net, train_iter, test_iter, num_epochs, lr_improved, device, scheduler)

# 绘制三个模型的对比图
plt.figure(figsize=(15, 10))

# 损失曲线对比
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss, 'b-', label='LeNet Training Loss')
plt.plot(range(1, num_epochs+1), train_loss_bn, 'r--', label='LeNet+BN Training Loss')
plt.plot(range(1, num_epochs+1), train_loss_improved, 'g-.', label='Improved LeNet Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs+1), test_loss, 'b-', label='LeNet Test Loss')
plt.plot(range(1, num_epochs+1), test_loss_bn, 'r--', label='LeNet+BN Test Loss')
plt.plot(range(1, num_epochs+1), test_loss_improved, 'g-.', label='Improved LeNet Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss Comparison')
plt.legend()

# 准确率曲线对比
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs+1), train_acc, 'b-', label='LeNet Training Accuracy')
plt.plot(range(1, num_epochs+1), train_acc_bn, 'r--', label='LeNet+BN Training Accuracy')
plt.plot(range(1, num_epochs+1), train_acc_improved, 'g-.', label='Improved LeNet Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs+1), test_acc, 'b-', label='LeNet Test Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_bn, 'r--', label='LeNet+BN Test Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_improved, 'g-.', label='Improved LeNet Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('./NNDL-Class/Project3/Result/lenet_all_comparison.png')

# 打印最终准确率对比
print("\nModel Performance Comparison:")
print("-" * 50)
print(f"Original LeNet Final Test Accuracy: {test_acc[-1]:.4f}")
print(f"LeNet with BatchNorm Final Test Accuracy: {test_acc_bn[-1]:.4f}")
print(f"Improved LeNet Final Test Accuracy: {test_acc_improved[-1]:.4f}")