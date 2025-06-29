import torch
from torch import nn
import torch.nn.functional as F

# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X - X.mean()

# 测试CenteredLayer
layer = CenteredLayer()
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
print("原始数据:", X)
Y = layer(X)
print("居中后:", Y)
print("均值为零:", Y.mean().item())

# 将自定义层集成到复杂模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print("\n网络输出均值:", Y.mean().item())

# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)

# 测试带参数的自定义层
dense = MyLinear(5, 3)
print("\n自定义线性层参数:")
print("权重形状:", dense.weight.shape)
print("偏置形状:", dense.bias.shape)

# 使用自定义层进行前向计算
X = torch.rand(2, 5)
print("\n输入形状:", X.shape)
print("输出形状:", dense(X).shape)

# 将自定义线性层集成到模型中
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print("\n复杂模型:", net)
print("输出形状:", net(torch.rand(2, 64)).shape)

# 访问模型参数
print("\n模型参数:")
for name, param in net.named_parameters():
    print(f"{name}, 形状: {param.shape}")
