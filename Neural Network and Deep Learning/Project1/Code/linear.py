import random
import torch
import matplotlib.pyplot as plt
import os
from d2l import torch as d2l

# 创建结果保存目录
save_dir = '/home/yyz/NNDL-Class/Project1/Result'
os.makedirs(save_dir, exist_ok=True)

# 1. 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 输出第一个样本的特征和标签
print('features:', features[0], '\nlabel:', labels[0])

# 绘制散点图观察并保存
plt.figure(figsize=(7, 5))
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.savefig(f'{save_dir}/linear_regression_data.png')
plt.close()

# 2. 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机打乱样本顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
# 获取第一个小批量数据并查看
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 3. 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 4. 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 5. 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 6. 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 7. 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# 记录每个epoch的损失
losses = []

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 计算损失
        # 反向传播计算梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 更新参数
    
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        mean_loss = float(train_l.mean())
        losses.append(mean_loss)
        print(f'epoch {epoch + 1}, loss {mean_loss:f}')

# 绘制损失曲线并保存
plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(f'{save_dir}/linear_regression_loss.png')
plt.close()

# 比较学习到的参数和真实参数
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
