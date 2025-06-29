import torch
import torch.nn as nn
import torch.optim as optim

# 1. 首先定义一个原始网络
class OriginalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络的各个层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 8 * 8, 10)  # 假设输入图像为32x32
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. 创建并初始化原始网络
original_net = OriginalNetwork()
print("原始网络结构:")
print(original_net)

# 3. 训练原始网络（这里只做简单示例）
# 创建一些随机数据
X = torch.randn(4, 3, 32, 32)
y = torch.randint(0, 10, (4,))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(original_net.parameters(), lr=0.01)

# 简单训练一步
optimizer.zero_grad()
outputs = original_net(X)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

# 4. 保存原始网络的参数
ORIGINAL_MODEL_PATH = "./Project2/Result/original_model.pth"
torch.save(original_net.state_dict(), ORIGINAL_MODEL_PATH)
print(f"原始模型参数已保存到 {ORIGINAL_MODEL_PATH}")

# 5. 定义一个新的网络架构，该架构将复用原始网络的前两层
class NewNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 复用原始网络的前两层（结构相同，但参数需要加载）
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 新的自定义层
        self.layer3_new = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),  # 不同于原始网络
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_new = nn.Linear(128 * 8 * 8, 20)  # 不同的输出类别数
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_new(x)
        x = x.view(x.size(0), -1)
        x = self.fc_new(x)
        return x

# 6. 创建新网络
new_net = NewNetwork()
print("\n新网络结构:")
print(new_net)

# 7. 加载原始网络的参数到新网络的前两层
# 首先加载原始网络的完整状态字典
original_state_dict = torch.load(ORIGINAL_MODEL_PATH)

# 创建一个新的状态字典，只包含我们想要的层
new_state_dict = {}
for name, param in original_state_dict.items():
    # 只复制layer1和layer2的参数
    if name.startswith('layer1') or name.startswith('layer2'):
        new_state_dict[name] = param

# 使用strict=False允许部分加载参数
new_net.load_state_dict(new_state_dict, strict=False)
print("\n已加载原始网络的前两层参数到新网络")

# 8. 验证参数复用是否成功
print("\n验证参数复用:")
# 检查原始网络和新网络的layer1第一个卷积层的权重是否相同
original_conv1_weight = original_net.layer1[0].weight
new_conv1_weight = new_net.layer1[0].weight
print("原始网络和新网络的layer1卷积层权重是否相同:", 
      torch.allclose(original_conv1_weight, new_conv1_weight))

# 9. 冻结复用的层，只训练新层
print("\n冻结复用的层，只训练新层:")
# 冻结前两层参数
for param in new_net.layer1.parameters():
    param.requires_grad = False
for param in new_net.layer2.parameters():
    param.requires_grad = False

# 验证参数是否被冻结
for name, param in new_net.named_parameters():
    print(f"{name}, requires_grad: {param.requires_grad}")

# 10. 只训练新层（示例）
optimizer_new = optim.SGD(filter(lambda p: p.requires_grad, new_net.parameters()), lr=0.01)
criterion_new = nn.CrossEntropyLoss()

# 创建一些随机数据
X_new = torch.randn(4, 3, 32, 32)
y_new = torch.randint(0, 20, (4,))  # 新网络有20个类别

# 简单训练一步
optimizer_new.zero_grad()
outputs_new = new_net(X_new)
loss_new = criterion_new(outputs_new, y_new)
loss_new.backward()
optimizer_new.step()

print("\n新网络已训练一步，只更新了非冻结层的参数")

# 11. 检查冻结层的参数是否保持不变
print("\n检查冻结层参数是否保持不变:")
print("训练后，原始网络和新网络的layer1卷积层权重是否仍然相同:", 
      torch.allclose(original_net.layer1[0].weight, new_net.layer1[0].weight))

# 12. 完整的迁移学习流程（示例）
print("\n完整迁移学习流程示例:")
# 解冻所有层进行微调
for param in new_net.parameters():
    param.requires_grad = True

# 使用较小的学习率进行微调
optimizer_finetune = optim.SGD(new_net.parameters(), lr=0.001)

# 微调训练（示例）
optimizer_finetune.zero_grad()
outputs_finetune = new_net(X_new)
loss_finetune = criterion_new(outputs_finetune, y_new)
loss_finetune.backward()
optimizer_finetune.step()

print("所有层解冻后进行了微调训练")


