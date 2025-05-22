import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from d2l import torch as d2l

# 确保结果保存目录存在
save_dir = '/home/yyz/NNDL-Class/Project1/Result'
os.makedirs(save_dir, exist_ok=True)

# 1. 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="/home/yyz/NNDL-Class/Project1/Data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="/home/yyz/NNDL-Class/Project1/Data", train=False, transform=trans, download=True)

# 查看数据集大小
print(len(mnist_train), len(mnist_test))

# 查看图像形状
print(mnist_train[0][0].shape)

# 定义文本标签
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 定义可视化函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, save_path=None):
    """绘制图像列表并保存"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

# 展示训练集中的一些样本并保存
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, 
           titles=get_fashion_mnist_labels(y), 
           save_path=f'{save_dir}/fashion_mnist_samples.png')

# 2. 读取小批量
batch_size = 256
def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

# 测量读取时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

# 3. 整合所有组件
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/home/yyz/NNDL-Class/Project1/Data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/home/yyz/NNDL-Class/Project1/Data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

# 测试调整大小功能
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

# 保存一个调整大小后的样本图像
resized_imgs = X[:9]
show_images(resized_imgs.reshape(9, 64, 64), 3, 3, 
           titles=get_fashion_mnist_labels(y[:9]), 
           save_path=f'{save_dir}/fashion_mnist_resized.png')
