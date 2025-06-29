import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 创建MLP实例
net = MLP()
print("网络结构:", net)

# 访问参数
print("\n访问参数:")
# 访问隐藏层参数
print("隐藏层权重形状:", net.hidden.weight.shape)
print("隐藏层偏置形状:", net.hidden.bias.shape)
# 访问输出层参数
print("输出层权重形状:", net.output.weight.shape)
print("输出层偏置形状:", net.output.bias.shape)

# 通过named_parameters()访问所有参数
print("\n通过named_parameters()访问所有参数:")
for name, param in net.named_parameters():
    print(f"{name}, 形状: {param.shape}")

# 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# 应用初始化
net.apply(init_weights)
print("\n初始化后的参数:")
print("隐藏层权重:", net.hidden.weight.data[0][:5])  # 只打印部分数据
print("隐藏层偏置:", net.hidden.bias.data[:5])

# 测试前向传播
X = torch.rand(size=(2, 20))
output = net(X)
print("\n输入形状:", X.shape)
print("输出形状:", output.shape)
print("输出:", output)

# 检查参数梯度
loss = output.sum()
loss.backward()
print("\n参数梯度:")
print("隐藏层权重梯度形状:", net.hidden.weight.grad.shape)
print("输出层权重梯度形状:", net.output.weight.grad.shape)
