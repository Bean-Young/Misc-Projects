import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# 创建输入张量
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print("输入张量:")
print(X)

# 最大汇聚和平均汇聚
print("最大汇聚结果:")
print(pool2d(X, (2, 2)))
print("平均汇聚结果:")
print(pool2d(X, (2, 2), 'avg'))

# 使用PyTorch内置汇聚层
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("输入形状:", X.shape)
print("输入数据:")
print(X)

# 默认步幅与池化窗口大小相同
pool2d = nn.MaxPool2d(3)
print("3x3最大汇聚，默认步幅:")
print(pool2d(X))

# 指定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("3x3最大汇聚，填充=1，步幅=2:")
print(pool2d(X))

# 矩形汇聚窗口
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print("2x3最大汇聚，不同步幅和填充:")
print(pool2d(X))

# 多通道
X = torch.cat((X, X + 1), 1)
print("多通道输入形状:", X.shape)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("多通道汇聚结果:")
print(pool2d(X))
