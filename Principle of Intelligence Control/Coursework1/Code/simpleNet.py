import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义符号函数
def sign(x):
    return torch.sign(x)

# 初始化权重和偏置
w = torch.tensor([0.75, 0.5, -0.6], requires_grad=True)
bias = 1.0

# 训练数据
training_data = [
    (torch.tensor([1.0, 1.0]), 1),
    (torch.tensor([9.4, 6.4]), -1),
    (torch.tensor([2.5, 2.1]), 1),
    (torch.tensor([8.0, 7.7]), -1),
    (torch.tensor([0.5, 2.2]), 1),
    (torch.tensor([7.9, 8.4]), -1),
    (torch.tensor([7.0, 7.0]), -1),
    (torch.tensor([2.8, 0.8]), 1),
    (torch.tensor([1.2, 3.0]), 1),
    (torch.tensor([7.8, 6.1]), -1)
]

class SquareRootScheduler:
    def __init__(self, lr=0.2):
        self.lr = lr

    def __call__(self, epoch):
        return self.lr * pow(epoch + 1.0, -0.5)

class FactorScheduler:
    def __init__(self, factor=0.9, stop_factor_lr=1e-2, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

# 学习率调度器
# def learning_rate_scheduler(epoch):
#     base_lr = 0.2
#     return base_lr / (1 + 0.1 * epoch)

# 超参数
num_epochs = 100
learning_rate = 0.2
# scheduler = SquareRootScheduler(lr=learning_rate)
scheduler = FactorScheduler(base_lr=learning_rate)
# scheduler = CosineScheduler(max_update=10, base_lr=learning_rate)

# 记录每个epoch的损失和学习率
losses = []
learning_rates = []

# 开始训练
for epoch in range(num_epochs):
    correct_count = 0
    for x, d in training_data:
        # 添加偏置项到输入向量
        x_with_bias = torch.cat((x, torch.tensor([bias])))
        net = w[0] * x_with_bias[0] + w[1] * x_with_bias[1] + w[2] * x_with_bias[2]
        output = sign(net)
        if output == d:
            correct_count += 1
        else:
            # 计算误差
            error = d - output
            # 更新权重
            with torch.no_grad():
                w += learning_rate * error * x_with_bias
    # 更新学习率
    learning_rate = scheduler.__call__(epoch)
    losses.append(correct_count / len(training_data))
    learning_rates.append(learning_rate)
    print(f'Epoch {epoch+1}/{num_epochs}, Learning Rate: {learning_rate}')
    print(w)
    # 如果正确分类的样本数达到10个，则提前终止训练
    if correct_count >= 10:
        break

# 打印最终权重
print("Final weights:", w)
for x, d in training_data:
    # 添加偏置项到输入向量
    x_with_bias = torch.cat((x, torch.tensor([bias])))
    net = w[0] * x_with_bias[0] + w[1] * x_with_bias[1] + w[2] * x_with_bias[2]
    output = sign(net)
    print(output)

# 绘制损失和学习率的变化曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(learning_rates)), learning_rates, label='Learning Rate', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
