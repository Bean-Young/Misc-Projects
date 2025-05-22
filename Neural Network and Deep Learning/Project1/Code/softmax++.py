import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l

# 确保结果保存目录存在
save_dir = '/home/yyz/NNDL-Class/Project1/Result'
os.makedirs(save_dir, exist_ok=True)

# 确保数据保存目录存在
data_dir = '/home/yyz/NNDL-Class/Project1/Data'
os.makedirs(data_dir, exist_ok=True)

# 修改加载数据集函数以使用正确的数据路径
def load_data_fashion_mnist_custom(batch_size, resize=None):
    """下载Fashion-MNIST数据集到自定义路径，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                           num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                           num_workers=d2l.get_dataloader_workers()))

# 定义准确率计算函数
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 定义累加器
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 定义评估函数
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 修改自d2l的train_ch3函数，确保图表保存而不是显示
def train_ch3_concise(net, train_iter, test_iter, loss, num_epochs, trainer, save_path=None):
    """训练模型并保存结果图表"""
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 训练
        net.train()
        metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        
        # 测试
        test_acc = evaluate_accuracy(net, test_iter)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'epoch {epoch+1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    
    # 绘制训练过程图表并保存
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs, label='train acc')
    plt.plot(epochs, test_accs, label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.legend()
    plt.grid()
    plt.title(f'Training Results - {save_path.split("/")[-1].replace(".png", "")}')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return train_losses, train_accs, test_accs

# 定义获取Fashion-MNIST标签的函数
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 预测并保存结果
def predict_fashion_mnist(net, test_iter, n=6, save_path=None):
    """预测Fashion-MNIST测试数据标签并保存结果"""
    X, y = next(iter(test_iter))
    
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    
    plt.figure(figsize=(12, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28).detach().numpy())
        plt.axis('off')
        plt.title(titles[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

# 定义初始化权重函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 练习1: 尝试调整超参数，例如批量大小、迭代周期数和学习率，并查看结果

# 基础版本 - 批量大小256，学习率0.1，迭代周期10
print("实验1: 基础版本 - 批量大小256，学习率0.1，迭代周期10")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net1 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net1.apply(init_weights)
trainer1 = torch.optim.SGD(net1.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss(reduction='none')
results1 = train_ch3_concise(net1, train_iter, test_iter, loss, 10, trainer1, 
                           save_path=f'{save_dir}/exp1_base_bs256_lr0.1_ep10.png')
predict_fashion_mnist(net1, test_iter, n=6, 
                    save_path=f'{save_dir}/exp1_base_predictions.png')

# 实验2 - 增大批量大小到512
print("\n实验2: 增大批量大小 - 批量大小512，学习率0.1，迭代周期10")
batch_size = 512
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net2 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net2.apply(init_weights)
trainer2 = torch.optim.SGD(net2.parameters(), lr=0.1)
results2 = train_ch3_concise(net2, train_iter, test_iter, loss, 10, trainer2, 
                           save_path=f'{save_dir}/exp2_large_bs512_lr0.1_ep10.png')

# 实验3 - 减小批量大小到64
print("\n实验3: 减小批量大小 - 批量大小64，学习率0.1，迭代周期10")
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net3 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net3.apply(init_weights)
trainer3 = torch.optim.SGD(net3.parameters(), lr=0.1)
results3 = train_ch3_concise(net3, train_iter, test_iter, loss, 10, trainer3, 
                           save_path=f'{save_dir}/exp3_small_bs64_lr0.1_ep10.png')

# 实验4 - 增大学习率到0.5
print("\n实验4: 增大学习率 - 批量大小256，学习率0.5，迭代周期10")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net4 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net4.apply(init_weights)
trainer4 = torch.optim.SGD(net4.parameters(), lr=0.5)
results4 = train_ch3_concise(net4, train_iter, test_iter, loss, 10, trainer4, 
                           save_path=f'{save_dir}/exp4_large_lr0.5_bs256_ep10.png')

# 实验5 - 减小学习率到0.01
print("\n实验5: 减小学习率 - 批量大小256，学习率0.01，迭代周期10")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net5 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net5.apply(init_weights)
trainer5 = torch.optim.SGD(net5.parameters(), lr=0.01)
results5 = train_ch3_concise(net5, train_iter, test_iter, loss, 10, trainer5, 
                           save_path=f'{save_dir}/exp5_small_lr0.01_bs256_ep10.png')

# 练习2: 增加迭代周期的数量，观察过拟合现象

# 实验6 - 增加迭代周期数到50，观察过拟合
print("\n实验6: 增加迭代周期数 - 批量大小256，学习率0.1，迭代周期50")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net6 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net6.apply(init_weights)
trainer6 = torch.optim.SGD(net6.parameters(), lr=0.1)
results6 = train_ch3_concise(net6, train_iter, test_iter, loss, 50, trainer6, 
                           save_path=f'{save_dir}/exp6_overfit_ep50_bs256_lr0.1.png')

# 实验7 - 使用L2正则化解决过拟合问题
print("\n实验7: 使用L2正则化 - 批量大小256，学习率0.1，迭代周期50，权重衰减0.001")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net7 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net7.apply(init_weights)
trainer7 = torch.optim.SGD(net7.parameters(), lr=0.1, weight_decay=0.001)  # 添加L2正则化
results7 = train_ch3_concise(net7, train_iter, test_iter, loss, 50, trainer7, 
                           save_path=f'{save_dir}/exp7_l2reg_ep50_bs256_lr0.1_wd0.001.png')

# 实验8 - 使用早停（Early Stopping）解决过拟合
print("\n实验8: 使用早停（Early Stopping）- 批量大小256，学习率0.1")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net8 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net8.apply(init_weights)
trainer8 = torch.optim.SGD(net8.parameters(), lr=0.1)

# 实现早停
patience = 5  # 如果测试精度连续5个epoch没有提升，则停止训练
best_acc = 0
no_improve_count = 0
max_epochs = 100
early_stop_epoch = 0

train_losses = []
train_accs = []
test_accs = []

for epoch in range(max_epochs):
    # 训练
    net8.train()
    metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数
    for X, y in train_iter:
        y_hat = net8(X)
        l = loss(y_hat, y)
        trainer8.zero_grad()
        l.mean().backward()
        trainer8.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    
    # 测试
    test_acc = evaluate_accuracy(net8, test_iter)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f'epoch {epoch+1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    
    # 早停逻辑
    if test_acc > best_acc:
        best_acc = test_acc
        no_improve_count = 0
        # 保存最佳模型参数（实际应用中）
        # torch.save(net8.state_dict(), f'{save_dir}/best_model.pth')
    else:
        no_improve_count += 1
    
    if no_improve_count >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        early_stop_epoch = epoch + 1
        break

# 绘制训练过程图表并保存
plt.figure(figsize=(10, 6))
epochs = list(range(1, len(train_losses) + 1))
plt.plot(epochs, train_losses, label='train loss')
plt.plot(epochs, train_accs, label='train acc')
plt.plot(epochs, test_accs, label='test acc')
if early_stop_epoch > 0:
    plt.axvline(x=early_stop_epoch, color='r', linestyle='--', 
                label=f'Early stopping (epoch {early_stop_epoch})')
plt.xlabel('epoch')
plt.ylabel('metric')
plt.legend()
plt.grid()
plt.title('Training with Early Stopping')
plt.savefig(f'{save_dir}/exp8_early_stopping.png')
plt.close()

# 实验9 - 使用Dropout解决过拟合问题
print("\n实验9: 使用Dropout - 批量大小256，学习率0.1，迭代周期50，Dropout比率0.5")
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)
net9 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 添加Dropout层
    nn.Linear(256, 10)
)
net9.apply(init_weights)
trainer9 = torch.optim.SGD(net9.parameters(), lr=0.1)
results9 = train_ch3_concise(net9, train_iter, test_iter, loss, 50, trainer9, 
                           save_path=f'{save_dir}/exp9_dropout_ep50_bs256_lr0.1.png')

# 比较所有实验结果 - 比较最终测试精度
plt.figure(figsize=(12, 8))
plt.bar(['Base', 'BS=512', 'BS=64', 'LR=0.5', 'LR=0.01', 'Epochs=50', 'L2 Reg', 'Early Stop', 'Dropout'],
        [results1[2][-1], results2[2][-1], results3[2][-1], results4[2][-1], results5[2][-1], 
         results6[2][-1], results7[2][-1], test_accs[-1], results9[2][-1]])
plt.ylabel('Final Test Accuracy')
plt.title('Comparison of Final Test Accuracy Across Experiments')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{save_dir}/all_experiments_comparison.png')
plt.close()

# 分析练习2的过拟合问题 - 比较标准训练与使用正则化方法
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(results6[2]) + 1), results6[1], label='Train Acc (No Reg)')
plt.plot(range(1, len(results6[2]) + 1), results6[2], label='Test Acc (No Reg)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Without Regularization (50 epochs)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(results7[2]) + 1), results7[1], label='Train Acc (L2 Reg)')
plt.plot(range(1, len(results7[2]) + 1), results7[2], label='Test Acc (L2 Reg)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('With L2 Regularization (50 epochs)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f'{save_dir}/overfitting_analysis.png')
plt.close()

# 保存实验结论摘要
with open(f'{save_dir}/experiment_results_summary.txt', 'w') as f:
    f.write("Fashion-MNIST Softmax回归实验结果摘要\n")
    f.write("======================================\n\n")
    
    f.write("练习1: 超参数调整实验\n")
    f.write("-----------------------\n")
    f.write(f"实验1 (基础): 批量大小=256, 学习率=0.1, 迭代周期=10, 最终测试精度={results1[2][-1]:.4f}\n")
    f.write(f"实验2 (大批量): 批量大小=512, 学习率=0.1, 迭代周期=10, 最终测试精度={results2[2][-1]:.4f}\n")
    f.write(f"实验3 (小批量): 批量大小=64, 学习率=0.1, 迭代周期=10, 最终测试精度={results3[2][-1]:.4f}\n")
    f.write(f"实验4 (大学习率): 批量大小=256, 学习率=0.5, 迭代周期=10, 最终测试精度={results4[2][-1]:.4f}\n")
    f.write(f"实验5 (小学习率): 批量大小=256, 学习率=0.01, 迭代周期=10, 最终测试精度={results5[2][-1]:.4f}\n\n")
    
    f.write("分析:\n")
    f.write("- 批量大小影响: 小批量通常提供更好的泛化性能，但训练时间更长\n")
    f.write("- 学习率影响: 过大的学习率可能导致不稳定，过小的学习率收敛慢\n\n")
    
    f.write("练习2: 过拟合问题与解决方案\n")
    f.write("----------------------------\n")
    f.write(f"实验6 (过拟合): 批量大小=256, 学习率=0.1, 迭代周期=50, 最终测试精度={results6[2][-1]:.4f}\n")
    f.write(f"实验7 (L2正则化): 批量大小=256, 学习率=0.1, 迭代周期=50, 权重衰减=0.001, 最终测试精度={results7[2][-1]:.4f}\n")
    f.write(f"实验8 (早停): 批量大小=256, 学习率=0.1, 停止于第{early_stop_epoch}个迭代周期, 最终测试精度={test_accs[-1]:.4f}\n")
    f.write(f"实验9 (Dropout): 批量大小=256, 学习率=0.1, 迭代周期=50, Dropout率=0.5, 最终测试精度={results9[2][-1]:.4f}\n\n")
    