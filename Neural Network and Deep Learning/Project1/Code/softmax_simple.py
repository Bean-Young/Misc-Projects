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

# 1. 设置超参数和自定义加载数据集函数
batch_size = 256

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

# 使用自定义函数加载数据集
train_iter, test_iter = load_data_fashion_mnist_custom(batch_size)

# 2. 初始化模型参数
# PyTorch不会隐式地调整输入的形状，因此定义展平层(flatten)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 3. 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 4. 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 5. 训练模型
num_epochs = 10

# 修改自d2l的train_ch3函数，确保图表保存而不是显示
def train_ch3_concise(net, train_iter, test_iter, loss, num_epochs, trainer, save_path=None):
    """训练模型并保存结果图表"""
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 定义准确率计算函数
    def accuracy(y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())
    
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
    if save_path:
        plt.savefig(save_path)
        plt.close()

# 训练模型并保存结果
train_ch3_concise(net, train_iter, test_iter, loss, num_epochs, trainer, 
                 save_path=f'{save_dir}/softmax_concise_training.png')

# 预测并保存结果
def predict_fashion_mnist(net, test_iter, n=6, save_path=None):
    """预测Fashion-MNIST测试数据标签并保存结果"""
    X, y = next(iter(test_iter))
    
    # 获取Fashion-MNIST标签
    def get_fashion_mnist_labels(labels):
        """返回Fashion-MNIST数据集的文本标签"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                      'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]
    
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

predict_fashion_mnist(net, test_iter, n=6, save_path=f'{save_dir}/softmax_concise_predictions.png')

# 练习2：尝试调整超参数并保存结果
# 增加L2正则化来解决过拟合问题
net_reg = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net_reg.apply(init_weights)
trainer_reg = torch.optim.SGD(net_reg.parameters(), lr=0.1, weight_decay=0.001)  # 添加L2正则化

num_epochs_reg = 30
train_ch3_concise(net_reg, train_iter, test_iter, loss, num_epochs_reg, trainer_reg, 
                 save_path=f'{save_dir}/softmax_concise_regularized.png')
