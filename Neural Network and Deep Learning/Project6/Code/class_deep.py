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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 数据路径
data_dir = './NNDL-Class/Project6/Data/dog-breed-identification'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
labels_file = os.path.join(data_dir, 'labels.csv')

# 结果保存路径
result_dir = './NNDL-Class/Project6/Result/class'
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


# 2. 数据加载和预处理 - 使用更丰富的数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))
    ], p=0.3),
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
train_size = int(0.85 * len(train_ds))  # 85% 用于训练
valid_size = len(train_ds) - train_size  # 15% 用于验证

train_data, valid_data = torch.utils.data.random_split(train_ds, [train_size, valid_size])

# 使用较小的批量大小以适应更深的模型
batch_size = 32
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
        image = Image.open(img_name).convert('RGB')  # 确保RGB格式
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # 返回文件名用于预测结果


# 创建 test_images 文件夹（如果不存在）
test_images_dir = os.path.join(test_dir, 'test_images')
os.makedirs(test_images_dir, exist_ok=True)

# 移动 test 文件夹中的所有文件到 test_images
if not os.listdir(test_images_dir):
    print("Organizing test set...")
    for filename in os.listdir(test_dir):
        file_path = os.path.join(test_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.jpg'):
            shutil.move(file_path, os.path.join(test_images_dir, filename))
else:
    print("Test set already organized.")

test_ds = TestDataset(test_images_dir, transform=transform_test)
test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

def get_net(model_name='resnet50', freeze_backbone=True, devices=None):
    # 根据模型名称加载预训练模型
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
    
    # 替换输出层以适应120个类别
    if 'resnet' in model_name:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 120)
        )
   
    model = model.to(devices[0])
    
    # 冻结特征提取层
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # 解冻最后两个阶段进行微调
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'features.7' in name or 'features.8' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # 打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    return model

def train_model(model_name='resnet50', num_epochs=30, lr=0.001, wd=1e-4, 
                freeze_backbone=True, scheduler_type='plateau', devices=None):
    # 创建模型特定的结果目录
    model_result_dir = os.path.join(result_dir, model_name)
    os.makedirs(model_result_dir, exist_ok=True)
    
    # 获取当前时间戳用于保存唯一结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Training {model_name} for {num_epochs} epochs with lr={lr}, wd={wd}")
    
    # 加载模型
    net = get_net(model_name, freeze_backbone, devices)
    
    # 优化器 - 使用AdamW优化器
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
    
    # 学习率调度器
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    
    # 训练跟踪
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'lr': []
    }
    
    best_valid_acc = 0.0
    early_stop_counter = 0
    early_stop_patience = 7
    
    start_time = time.time()
    
    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + 1
        net.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # 训练阶段
        for inputs, labels in tqdm(train_iter, desc=f'Epoch {epoch}/{num_epochs} Training'):
            inputs, labels = inputs.to(devices[0]), labels.to(devices[0])
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        # 验证阶段
        valid_loss, valid_acc, valid_preds, valid_labels = evaluate_loss(valid_iter, net, devices)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(valid_acc)
            else:
                scheduler.step()
        
        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['lr'].append(current_lr)
        
        # 打印统计信息
        print(f'Epoch {epoch}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
              f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} | '
              f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            early_stop_counter = 0
            save_model(net, f'{model_name}_best_model.pth', model_result_dir)
            print(f'Best model saved with validation accuracy: {best_valid_acc:.4f}')
            
            # 保存验证集的预测结果用于后续分析
            save_validation_results(valid_preds, valid_labels, valid_data, 
                                   f'{model_name}_best_val_preds.csv', model_result_dir)
        else:
            early_stop_counter += 1
            print(f'No improvement for {early_stop_counter}/{early_stop_patience} epochs')
    
        # 早停检查
        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch} after {early_stop_patience} epochs without improvement')
            break
    
    # 训练完成后保存最终结果
    training_time = time.time() - start_time
    print(f'Training completed in {training_time//60:.0f}m {training_time%60:.0f}s')
    
    # 保存最终模型
    save_model(net, f'{model_name}_final_model.pth', model_result_dir)
    
    # 保存训练曲线
    save_training_curve(history, f'{model_name}_final_training_curve.png', model_result_dir)
    
    # 保存混淆矩阵
    _, _, valid_preds, valid_labels = evaluate_loss(valid_iter, net, devices)
    save_confusion_matrix(valid_preds, valid_labels, valid_data, 
                         f'{model_name}_final_confusion_matrix.png', model_result_dir)
    
    # 保存分类报告
    save_classification_report(valid_preds, valid_labels, valid_data, 
                              f'{model_name}_classification_report.txt', model_result_dir)
    
    # 保存训练历史
    save_training_history(history, f'{model_name}_training_history.csv', model_result_dir)
    
    # 在测试集上进行预测
    save_predictions(net, test_iter, f'{model_name}_submission.csv', model_result_dir, devices)
    
    return net, history

def evaluate_loss(data_iter, net, devices):
    net.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_iter, desc='Evaluating'):
            inputs, labels = inputs.to(devices[0]), labels.to(devices[0])
            
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item(), np.array(all_preds), np.array(all_labels)

def save_training_curve(history, filename, save_dir):
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label="Train Loss")
    plt.plot(history['epoch'], history['valid_loss'], 'r-', label="Valid Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['epoch'], history['train_acc'], 'b-', label="Train Accuracy")
    plt.plot(history['epoch'], history['valid_acc'], 'r-', label="Valid Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    
    # 最佳准确率标记
    best_acc = max(history['valid_acc'])
    best_epoch = history['epoch'][np.argmax(history['valid_acc'])]
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.5, f'Best Validation Accuracy: {best_acc:.4f}\nat Epoch: {best_epoch}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def save_confusion_matrix(preds, labels, dataset, filename, save_dir):
    # 获取类别名称
    class_names = list(dataset.dataset.class_to_idx.keys())
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def save_classification_report(preds, labels, dataset, filename, save_dir):
    class_names = list(dataset.dataset.class_to_idx.keys())
    report = classification_report(labels, preds, target_names=class_names)
    
    with open(os.path.join(save_dir, filename), 'w') as f:
        f.write(report)

def save_training_history(history, filename, save_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, filename), index=False)

def save_validation_results(preds, labels, dataset, filename, save_dir):
    class_names = list(dataset.dataset.class_to_idx.keys())
    true_classes = [class_names[i] for i in labels]
    pred_classes = [class_names[i] for i in preds]
    
    # 获取原始图像路径
    image_paths = [dataset.dataset.samples[i][0] for i in dataset.indices]
    
    df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': true_classes,
        'pred_label': pred_classes,
        'correct': [t == p for t, p in zip(true_classes, pred_classes)]
    })
    
    df.to_csv(os.path.join(save_dir, filename), index=False)

def save_model(net, filename, save_dir):
    torch.save(net.state_dict(), os.path.join(save_dir, filename))
    print(f'Model saved to {os.path.join(save_dir, filename)}')

def save_predictions(net, test_iter, filename, save_dir, devices):
    net.eval()
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_iter, desc='Predicting Test Set'):
            images = images.to(devices[0])
            outputs = net(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend([fname.replace('.jpg', '') for fname in filenames])
    
    # 创建提交文件
    class_names = sorted(os.listdir(train_dir))
    submission_df = pd.DataFrame(all_probs, columns=class_names)
    submission_df.insert(0, 'id', all_ids)
    submission_df.to_csv(os.path.join(save_dir, filename), index=False)
    print(f'Predictions saved to {os.path.join(save_dir, filename)}')

# 获取设备
devices = d2l.try_all_gpus()
print(f"Using devices: {devices}")

# 训练不同模型并保存结果
models_to_train = [
    {'model_name': 'resnet50', 'num_epochs': 40, 'lr': 0.001, 'freeze_backbone': False, 'scheduler_type': 'plateau'},
    {'model_name': 'resnet101', 'num_epochs': 40, 'lr': 0.0005, 'freeze_backbone': False, 'scheduler_type': 'plateau'},
]

all_results = {}

for config in models_to_train:
    print(f"\n{'='*50}")
    print(f"Training {config['model_name']} model")
    print(f"{'='*50}")
    
    model, history = train_model(
        model_name=config['model_name'],
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        wd=1e-4,
        freeze_backbone=config['freeze_backbone'],
        scheduler_type=config['scheduler_type'],
        devices=devices
    )
    
    all_results[config['model_name']] = {
        'history': history,
        'best_val_acc': max(history['valid_acc'])
    }
    
    # 释放内存
    del model
    torch.cuda.empty_cache()

# 比较不同模型的结果
print("\nModel Comparison Results:")
for model_name, result in all_results.items():
    print(f"{model_name}: Best Validation Accuracy = {result['best_val_acc']:.4f}")

# 绘制所有模型的验证准确率比较
plt.figure(figsize=(12, 8))
for model_name, result in all_results.items():
    plt.plot(result['history']['epoch'], result['history']['valid_acc'], 
             label=f"{model_name} (best: {result['best_val_acc']:.4f})")

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Model Comparison: Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'model_comparison.png'), dpi=300)
plt.close()