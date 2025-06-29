import torch
from torch import nn

# 定义并行块
class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        
    def forward(self, X):
        return self.net1(X) + self.net2(X)

# 测试ParallelBlock
# 创建两个简单的网络
net1 = nn.Sequential(nn.Linear(20, 30), nn.ReLU(), nn.Linear(30, 10))
net2 = nn.Sequential(nn.Linear(20, 40), nn.ReLU(), nn.Linear(40, 10))

# 将它们组合成一个ParallelBlock
parallel_net = ParallelBlock(net1, net2)

# 测试
X = torch.rand(2, 20)
output = parallel_net(X)
print("输入形状:", X.shape)
print("输出形状:", output.shape)
print("输出:", output)

# 验证结果是否为两个网络输出的和
output1 = net1(X)
output2 = net2(X)
print("验证输出是否为两个网络输出的和:", torch.allclose(output, output1 + output2))

# 检查模型参数是否被正确注册
print("\n模型结构:")
print(parallel_net)

# 检查参数数量
total_params = sum(p.numel() for p in parallel_net.parameters())
net1_params = sum(p.numel() for p in net1.parameters())
net2_params = sum(p.numel() for p in net2.parameters())
print(f"总参数数量: {total_params}")
print(f"net1参数数量: {net1_params}")
print(f"net2参数数量: {net2_params}")
print(f"验证参数数量: {total_params == net1_params + net2_params}")

