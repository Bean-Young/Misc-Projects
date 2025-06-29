import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from d2l import torch as d2l
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 数据路径
data_dir = './NNDL-Class/Project6/Data/dog-breed-identification'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
labels_file = os.path.join(data_dir, 'labels.csv')

# 结果保存路径
result_dir = './NNDL-Class/Project6/Result'
os.makedirs(result_dir, exist_ok=True)

# 1. 数据整理：根据标签文件将图像移动到相应的子文件夹中
labels_df = pd.read_csv(labels_file)

# 创建目标文件夹，如果它不存在
for breed in labels_df['breed'].unique():
    os.makedirs(os.path.join(train_dir, breed), exist_ok=True)

# 将图片移动到相应的文件夹
for _, row in labels_df.iterrows():
    image_name = row['id'] + '.jpg'
    breed = row['breed']
    src_path = os.path.join(train_dir, image_name)
    dst_path = os.path.join(train_dir, breed, image_name)
    
    # 如果图片存在，则移动
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)


# 2. 数据加载和预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练集和验证集
train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
train_size = int(0.9 * len(train_ds))  # 90% 用于训练
valid_size = len(train_ds) - train_size  # 10% 用于验证

train_data, valid_data = torch.utils.data.random_split(train_ds, [train_size, valid_size])

# 减小批量大小以适配GPU内存
train_iter = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
valid_iter = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4)

# 加载测试集图像（无需标签）
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # 返回文件名用于预测结果


# 创建 test_images 文件夹（如果不存在）
test_images_dir = os.path.join(test_dir, 'test_images')
os.makedirs(test_images_dir, exist_ok=True)

# 移动 test 文件夹中的所有文件到 test_images
for filename in os.listdir(test_dir):
    file_path = os.path.join(test_dir, filename)
    if os.path.isfile(file_path) and filename.endswith('.jpg'):  # 确保是jpg文件
        shutil.move(file_path, os.path.join(test_images_dir, filename))

test_ds = TestDataset(test_images_dir, transform=transform_test)
test_iter = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

def get_net(devices):
    # 加载预训练的ResNet34模型
    finetune_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # 替换ResNet34的输出层，调整为120个类别
    # 添加更多层以提高模型容量
    finetune_net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(finetune_net.fc.in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 120)  # 120类狗的品种
    )
    
    finetune_net = finetune_net.to(devices[0])

    # 解冻最后两个残差块以进行微调
    for name, param in finetune_net.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return finetune_net

# 评估模型损失和准确率
def evaluate_loss(data_iter, net, devices):
    net.eval()
    l_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for features, labels in tqdm(data_iter, desc='Evaluating Validation'):
            features, labels = features.to(devices[0]), labels.to(devices[0])
            output = net(features)
            l = nn.CrossEntropyLoss()(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            l_sum += l.item()
            acc_sum += acc
            n += labels.shape[0]
    return l_sum / len(data_iter), acc_sum / n

# 训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 只优化需要梯度的参数
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    trainer = optim.SGD(params_to_optimize, lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = StepLR(trainer, step_size=lr_period, gamma=lr_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    epoch = []
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    best_valid_acc = 0.0
    
    for epoch_idx in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)  # 用于保存损失和准确率
        
        for features, labels in tqdm(train_iter, desc=f'Epoch {epoch_idx+1}/{num_epochs} Training'):
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            loss = loss_fn(output, labels)
            loss.backward()
            trainer.step()
            
            # 更新指标
            metric.add(loss.item(), (output.argmax(dim=1) == labels).sum().item(), labels.shape[0])
        
        # 更新学习率
        scheduler.step()
        current_lr = trainer.param_groups[0]['lr']
        
        # 计算本epoch指标
        epoch.append(epoch_idx + 1)
        train_loss.append(metric[0] / metric[2])
        train_acc.append(metric[1] / metric[2])
        
        # 验证集评估
        valid_loss_val, valid_acc_val = evaluate_loss(valid_iter, net, devices)
        valid_loss.append(valid_loss_val)
        valid_acc.append(valid_acc_val)
        
        print(f'Epoch {epoch_idx+1}/{num_epochs}: '
              f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, '
              f'Valid Loss: {valid_loss[-1]:.4f}, Valid Acc: {valid_acc[-1]:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if valid_acc_val > best_valid_acc:
            best_valid_acc = valid_acc_val
            save_model(net, 'best_model.pth')
            print(f'Best model saved with validation accuracy: {best_valid_acc:.4f}')
    
    # 保存最终训练曲线图像
    save_training_curve(epoch, train_loss, train_acc, valid_loss, valid_acc, 'final_training_curve.png')
    return net

# 保存训练曲线的函数
def save_training_curve(epoch, train_loss, train_acc, valid_loss, valid_acc, filename):
    plt.figure(figsize=(12, 10))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(epoch, train_loss, 'b-', label="Train Loss")
    plt.plot(epoch, valid_loss, 'r-', label="Valid Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epoch, train_acc, 'b-', label="Train Accuracy")
    plt.plot(epoch, valid_acc, 'r-', label="Valid Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=300)
    plt.close()

# 保存模型
def save_model(net, filename):
    torch.save(net.state_dict(), os.path.join(result_dir, filename))
    print(f'Model saved to {filename}')

# 保存预测结果
def save_predictions(net, test_iter, filename):
    net.eval()
    preds = []
    ids = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_iter, desc='Predicting Test Set'):
            images = images.to(devices[0])
            outputs = net(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds.extend(probs.cpu().numpy())
            ids.extend([fname.replace('.jpg', '') for fname in filenames])
    
    # 创建提交文件
    classes = sorted(os.listdir(train_dir))
    submission_df = pd.DataFrame(preds, columns=classes)
    submission_df.insert(0, 'id', ids)
    submission_df.to_csv(os.path.join(result_dir, filename), index=False)
    print(f'Predictions saved to {filename}')

# 获取设备
devices = d2l.try_all_gpus()
print(f"Using devices: {devices}")
net = get_net(devices)

# 训练模型并保存训练曲线
# 增加学习率，添加学习率调度
net = train(net, train_iter, valid_iter, num_epochs=20, lr=0.01, wd=1e-4, 
            devices=devices, lr_period=5, lr_decay=0.5)

# 保存最终模型
save_model(net, 'final_model.pth')

# 预测并保存测试集结果
save_predictions(net, test_iter, 'submission.csv')