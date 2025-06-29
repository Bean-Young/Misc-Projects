import torch
from torch import nn

# 构建包含共享参数层的多层感知机
shared_layer = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared_layer,
    nn.ReLU(),
    shared_layer,
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 检查网络结构
print("网络结构:")
print(net)

# 初始化一些数据
X = torch.rand(size=(2, 4))

# 训练前检查参数
print("\n训练前检查参数:")
print("第3层和第5层权重是否相同:", torch.all(net[2].weight == net[4].weight))
print("第3层和第5层权重的内存地址是否相同:", net[2].weight.data_ptr() == net[4].weight.data_ptr())

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 简单训练过程
y = torch.rand(size=(2, 1))
print("\n训练过程:")
for i in range(5):
    optimizer.zero_grad()
    y_hat = net(X)
    loss = loss_fn(y_hat, y)
    loss.backward()
    
    # 检查梯度和参数
    print(f"\n第{i+1}次迭代:")
    print("第3层和第5层权重是否相同:", torch.all(net[2].weight == net[4].weight))
    if net[2].weight.grad is not None and net[4].weight.grad is not None:
        print("第3层梯度:", net[2].weight.grad[0][:3])
        print("第5层梯度:", net[4].weight.grad[0][:3])
        print("第3层和第5层梯度是否相同:", torch.all(net[2].weight.grad == net[4].weight.grad))
    
    optimizer.step()
    
    # 检查更新后的参数
    print("更新后第3层权重:", net[2].weight[0][:3])
    print("更新后第5层权重:", net[4].weight[0][:3])

