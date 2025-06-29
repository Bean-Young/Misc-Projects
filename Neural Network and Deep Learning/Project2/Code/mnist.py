import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# 设置全局中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号乱码

torch.manual_seed(42)

# 数据加载和预处理
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root='./Project1/Data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./Project1/Data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义基本模型类
class MLP(nn.Module):
    def __init__(self, activation=nn.ReLU(), dropout_rate=0.0):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练函数
def train_model(model, train_loader, test_loader, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 测试精度
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return train_losses, train_accs, test_accs


# 实验1：使用ReLU激活函数
model_relu = MLP(activation=nn.ReLU())
losses_relu, train_accs_relu, test_accs_relu = train_model(model_relu, train_loader, test_loader)

# 绘制损失和准确率曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses_relu)
plt.title('训练损失')
plt.xlabel('轮次')
plt.ylabel('损失')

plt.subplot(1, 3, 2)
plt.plot(train_accs_relu)
plt.title('训练准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')

plt.subplot(1, 3, 3)
plt.plot(test_accs_relu)
plt.title('测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')

plt.tight_layout()
plt.savefig('./Project2/Result/fashion_mnist_relu.png')
plt.close()  # 关闭图形而不是显示

# 实验2和3：不同激活函数比较
model_sigmoid = MLP(activation=nn.Sigmoid())
losses_sigmoid, train_accs_sigmoid, test_accs_sigmoid = train_model(model_sigmoid, train_loader, test_loader)

model_tanh = MLP(activation=nn.Tanh())
losses_tanh, train_accs_tanh, test_accs_tanh = train_model(model_tanh, train_loader, test_loader)

# 绘制比较曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses_relu, label='ReLU')
plt.plot(losses_sigmoid, label='Sigmoid')
plt.plot(losses_tanh, label='Tanh')
plt.title('训练损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accs_relu, label='ReLU')
plt.plot(train_accs_sigmoid, label='Sigmoid')
plt.plot(train_accs_tanh, label='Tanh')
plt.title('训练准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_accs_relu, label='ReLU')
plt.plot(test_accs_sigmoid, label='Sigmoid')
plt.plot(test_accs_tanh, label='Tanh')
plt.title('测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.savefig('./Project2/Result/fashion_mnist_activation_comparison.png')
plt.close()  # 关闭图形而不是显示

# 实验4：Dropout比较
model_dropout = MLP(activation=nn.ReLU(), dropout_rate=0.3)
losses_dropout, train_accs_dropout, test_accs_dropout = train_model(model_dropout, train_loader, test_loader)

# 绘制比较曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses_relu, label='ReLU')
plt.plot(losses_dropout, label='ReLU + Dropout')
plt.title('训练损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accs_relu, label='ReLU')
plt.plot(train_accs_dropout, label='ReLU + Dropout')
plt.title('训练准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_accs_relu, label='ReLU')
plt.plot(test_accs_dropout, label='ReLU + Dropout')
plt.title('测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.savefig('./Project2/Result/fashion_mnist_dropout_comparison.png')
plt.close()  # 关闭图形而不是显示

# 综合比较图
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses_relu, label='ReLU')
plt.plot(losses_sigmoid, label='Sigmoid')
plt.plot(losses_tanh, label='Tanh')
plt.plot(losses_dropout, label='ReLU + Dropout')
plt.title('训练损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accs_relu, label='ReLU')
plt.plot(train_accs_sigmoid, label='Sigmoid')
plt.plot(train_accs_tanh, label='Tanh')
plt.plot(train_accs_dropout, label='ReLU + Dropout')
plt.title('训练准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_accs_relu, label='ReLU')
plt.plot(test_accs_sigmoid, label='Sigmoid')
plt.plot(test_accs_tanh, label='Tanh')
plt.plot(test_accs_dropout, label='ReLU + Dropout')
plt.title('测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.savefig('./Project2/Result/fashion_mnist_comparison.png')
plt.close()  # 关闭图形而不是显示
