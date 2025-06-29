import torch
from torch import nn

class ParameterizedTensorDot(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 创建一个形状为(in_features, in_features, in_features)的参数张量
        self.W = nn.Parameter(torch.rand(in_features, in_features, in_features))
    
    def forward(self, x):
        # x的形状为(batch_size, in_features)
        batch_size = x.shape[0]
        
        # 使用einsum计算张量积
        # 计算 y_k = sum_{i,j} W_{ijk} * x_i * x_j
        # 先计算 x_i * W_{ijk}
        tmp = torch.einsum('bi,ijk->bjk', x, self.W)
        
        # 再计算 (x_i * W_{ijk}) * x_j
        y = torch.einsum('bjk,bj->bk', tmp, x)
        
        return y

# 测试参数化张量积层
in_features = 3
layer = ParameterizedTensorDot(in_features)

# 检查参数形状
print("参数W的形状:", layer.W.shape)

# 生成一个小批量样本
batch_size = 2
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("\n输入x的形状:", x.shape)

# 前向传播
output = layer(x)
print("输出的形状:", output.shape)
print("输出:", output)

# 使用循环验证计算结果的正确性
def manual_tensor_dot(W, x):
    batch_size, in_features = x.shape
    output = torch.zeros(batch_size, in_features)
    
    for b in range(batch_size):
        for k in range(in_features):
            sum_value = 0.0
            for i in range(in_features):
                for j in range(in_features):
                    sum_value += W[i, j, k] * x[b, i] * x[b, j]
            output[b, k] = sum_value
    
    return output

# 使用循环方法计算
manual_output = manual_tensor_dot(layer.W, x)
print("\n手动计算的输出:", manual_output)

# 验证两种方法计算结果是否一致
print("验证结果:", torch.allclose(output, manual_output, rtol=1e-5))

# 在更大的模型中使用自定义层
net = nn.Sequential(
    nn.Linear(10, 3),
    nn.ReLU(),
    ParameterizedTensorDot(3),
    nn.Linear(3, 1)
)

# 测试网络
test_input = torch.randn(5, 10)
test_output = net(test_input)
print("\n完整网络输出形状:", test_output.shape)

# 检查参数和梯度
print("\n参数检查:")
for name, param in net.named_parameters():
    print(f"{name}, 形状: {param.shape}")

# 计算梯度
loss = test_output.sum()
loss.backward()

# 验证张量积层的梯度是否正确计算
print("\n梯度检查:")
print("张量积层W的梯度形状:", net[2].W.grad.shape)
print("梯度是否包含NaN:", torch.isnan(net[2].W.grad).any())
