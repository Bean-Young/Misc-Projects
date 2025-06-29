import torch
import torch.nn as nn
import torch.optim as optim
import os

# 创建一个简单的模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

# 初始化模型
net = MLP()
print("原始模型结构:", net)

# 生成一些随机数据进行前向传播
X = torch.randn(2, 20)
y = net(X)
print("前向传播结果形状:", y.shape)

# 保存模型参数
PATH = './Project2/Result/mlp.params'
torch.save(net.state_dict(), PATH)
print(f"模型参数已保存到 {PATH}")

# 创建一个新的网络实例
net2 = MLP()
# 加载参数
net2.load_state_dict(torch.load(PATH))
print("加载参数后的模型:", net2)

# 验证加载后的模型输出是否相同
y2 = net2(X)
print("原始输出和加载后输出是否相同:", torch.allclose(y, y2))

# 保存整个模型
PATH_WHOLE = './Project2/Result/mlp.pt'
torch.save(net, PATH_WHOLE)
print(f"整个模型已保存到 {PATH_WHOLE}")

# 加载整个模型
net3 = torch.load(PATH_WHOLE)
print("加载整个模型:", net3)

# 验证加载后的模型输出是否相同
y3 = net3(X)
print("原始输出和加载整个模型后输出是否相同:", torch.allclose(y, y3))

# 保存和加载模型参数字典和优化器状态
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 保存检查点（包含模型参数和优化器状态）
CHECKPOINT_PATH = './Project2/Result/checkpoint.pth'
checkpoint = {
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 1,
    'loss': 0.5
}
torch.save(checkpoint, CHECKPOINT_PATH)
print(f"检查点已保存到 {CHECKPOINT_PATH}")

# 加载检查点
checkpoint = torch.load(CHECKPOINT_PATH)
net4 = MLP()
net4.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.SGD(net4.parameters(), lr=0.001, momentum=0.9)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(f"从检查点加载 - 轮次: {epoch}, 损失: {loss}")
