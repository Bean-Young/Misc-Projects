import torch
import matplotlib.pyplot as plt
import os
from d2l import torch as d2l

# 确保结果保存目录存在
save_dir = '/home/yyz/NNDL-Class/Project1/Result'
os.makedirs(save_dir, exist_ok=True)

# 确保数据保存目录存在
data_dir = '/home/yyz/NNDL-Class/Project1/Data'
os.makedirs(data_dir, exist_ok=True)

# 1. 设置超参数和自定义加载数据集函数
batch_size = 256

# 修改加载数据集函数以使用正确的数据路径
def load_data_fashion_mnist_custom(batch_size, resize=None):
    """下载Fashion-MNIST数据集到自定义路径，然后将其加载到内存中"""
    import torchvision
    from torchvision import transforms
    from torch.utils import data
    
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

# 使用自定义函数加载数据集
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)

# 2. 初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 3. 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 4. 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 5. 定义损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 6. 定义精度计算函数
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 定义用于累加的实用程序类
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

# 7. 训练函数定义
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 修改训练函数，保存图表而非显示
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, save_path=None):
    """训练模型并保存结果图表"""
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        
        train_loss, train_acc = train_metrics
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
    if save_path:
        plt.savefig(save_path)
        plt.close()

# 8. 训练模型
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater, 
         save_path=f'{save_dir}/softmax_scratch_training.png')

# 9. 预测并保存结果
def predict_ch3(net, test_iter, n=6, save_path=None):
    """预测标签并保存结果"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    
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

predict_ch3(net, test_iter, n=6, save_path=f'{save_dir}/softmax_scratch_predictions.png')
