import torch
from torch import nn
from d2l import torch as d2l

# 为了方便起见，定义一个计算卷积层的函数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 创建输入张量
X = torch.rand(size=(8, 8))
print("输入形状:", X.shape)

# 应用填充以保持输出形状与输入相同
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
Y = comp_conv2d(conv2d, X)
print("填充=1，输出形状:", Y.shape)

# 不同形状的卷积核和不同的填充
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
Y = comp_conv2d(conv2d, X)
print("不同填充，输出形状:", Y.shape)

# 使用步幅减少输出形状
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
Y = comp_conv2d(conv2d, X)
print("步幅=2，输出形状:", Y.shape)

# 复杂步幅示例
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
Y = comp_conv2d(conv2d, X)
print("复杂步幅和填充，输出形状:", Y.shape)
