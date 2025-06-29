import torch
from torch import nn

# 创建一个具有单隐藏层的网络
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print("输入X形状:", X.shape)
print("输出形状:", net(X).shape)

# 访问参数
print("\n访问参数:")
print(net[0].weight.data)  # 第一层的权重
print(net[0].bias.data)    # 第一层的偏置

# 一次性访问所有参数
print("\n一次性访问所有参数:")
print("类型:", type(net[0].named_parameters()))
for name, param in net[0].named_parameters():
    print(f"{name}, 形状: {param.shape}, 数据类型: {param.dtype}")

# 访问所有层的所有参数
print("\n访问所有层的所有参数:")
print("类型:", type(net.named_parameters()))
for name, param in net.named_parameters():
    print(f"{name}, 形状: {param.shape}")

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print("\n复杂网络结构:")
print(rgnet)

# 查看复杂网络参数
print("\n复杂网络参数:")
print("参数名称长度:", len(list(rgnet.named_parameters())))
for name, param in rgnet.named_parameters():
    print(f"{name}, 形状: {param.shape}")

# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print("\n初始化后的权重:")
print(net[0].weight.data)
print(net[0].bias.data)

# 自定义初始化
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print("\n常量初始化后的权重:")
print(net[0].weight.data)

# 对不同块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print("\n不同初始化方法:")
print("第一层:", net[0].weight.data[0])
print("第三层:", net[2].weight.data[0])

# 自定义参数初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("使用自定义初始化")
        with torch.no_grad():
            m.weight.fill_(1/m.weight.numel())
            m.bias.fill_(0)

net.apply(my_init)
print("\n自定义初始化后:")
print(net[0].weight.data)

# 参数绑定
# 我们需要共享这两个层的权重参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                   shared, nn.ReLU(),
                   shared, nn.ReLU(),
                   nn.Linear(8, 1))
print("\n参数绑定:")
# 检查参数是否相同
print("第三层和第五层的权重是否相同:", id(net[2].weight) == id(net[4].weight))
print("第三层和第五层的权重是否相同:", net[2].weight.data_ptr() == net[4].weight.data_ptr())

# 训练模型
net[0].weight.data[0, 0] = 100
# 运行一次前向传播
print("\n运行前向传播:")
print("第一层第一个权重:", net[0].weight.data[0, 0])
Y = net(X)
print("输出:", Y)

# 多个层共享参数时的梯度
print("\n检查共享参数的梯度:")
# 初始化一些数据
X = torch.rand(size=(2, 4))
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                   shared, nn.ReLU(),
                   shared, nn.ReLU(),
                   nn.Linear(8, 1))

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 简单训练过程
y = torch.rand(size=(2, 1))
optimizer.zero_grad()
y_hat = net(X)
loss = loss_fn(y_hat, y)
loss.backward()

# 检查共享层的梯度
print("第三层的梯度:")
print(net[2].weight.grad)
print("第五层的梯度:")
print(net[4].weight.grad)
print("梯度是否相同:", torch.allclose(net[2].weight.grad, net[4].weight.grad))
