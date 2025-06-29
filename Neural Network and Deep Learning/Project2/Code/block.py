import torch
from torch import nn
from torch.nn import functional as F

# 自定义块
class MLP(nn.Module):
    # 声明层，这里声明了两个全连接层
    def __init__(self):
        # 调用父类的__init__来执行必要的初始化
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)     # 输出层
    
    # 定义前向传播函数
    def forward(self, X):
        # 注意这里使用ReLU激活函数
        return self.out(F.relu(self.hidden(X)))

# 测试MLP类
net = MLP()
X = torch.rand(2, 20)
print("MLP输出形状:", net(X).shape)

# 顺序块 - 使用nn.Sequential类
my_seq = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
# 测试Sequential类
print("Sequential输出形状:", my_seq(X).shape)

# 自定义MySequential类
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例
            # 我把它保存在'Module'类的成员变量_modules中
            # _modules是一个OrderedDict
            self._modules[str(idx)] = module
    
    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

# 测试MySequential类
my_seq = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print("MySequential输出形状:", my_seq(X).shape)

# 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# 测试FixedHiddenMLP类
net = FixedHiddenMLP()
print("FixedHiddenMLP输出:", net(X))

# 混合搭配各种组合块的方法
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))

# 组合各种块
chimera = nn.Sequential(
    NestMLP(),
    nn.Linear(16, 20),
    FixedHiddenMLP()
)

print("组合块输出:", chimera(X))

