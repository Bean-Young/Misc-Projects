import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import os
from d2l import torch as d2l

# 确保结果保存目录存在
save_dir = '/home/yyz/NNDL-Class/Project1/Result'
os.makedirs(save_dir, exist_ok=True)

# 1. 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2. 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 查看第一个批量
next(iter(data_iter))

# 3. 定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# 4. 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 5. 定义损失函数
loss = nn.MSELoss()

# 6. 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 7. 训练
num_epochs = 3
losses = []

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    losses.append(float(l))
    print(f'epoch {epoch + 1}, loss {l:f}')

# 绘制损失曲线并保存
plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Concise Implementation)')
plt.savefig(f'{save_dir}/linear_regression_concise_loss.png')
plt.close()

# 比较学习到的参数和真实参数
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
