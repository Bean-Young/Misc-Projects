import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from model.Unet import UNet
from model.UnetP import UNetPlus
from dataloader.loaderseg import SegDataset

# 参数
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
NUM_CLASSES = 4
SAVE_DIR = '/home/yyz/Unet-ML/result'
PRED_DIR = '/home/yyz/Unet-ML/data/0.7/predict_unet*'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# 路径
IMAGE_DIR = '/home/yyz/Unet-ML/data/0.7/image'
LABEL_DIR = '/home/yyz/Unet-ML/data/0.7/label'
TRAIN_LIST = '/home/yyz/Unet-ML/data/0.7/train.txt'
VAL_LIST = '/home/yyz/Unet-ML/data/0.7/val.txt'

# 读文件名
with open(TRAIN_LIST, 'r') as f:
    train_files = f.readlines()
with open(VAL_LIST, 'r') as f:
    val_files = f.readlines()

# 转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader

# 定义五种不同的数据增强策略
augment_transforms = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ToTensor()  # 原图，不加变化
    ]),
]

# 创建多个增强版本的数据集
augmented_datasets = [
    SegDataset(IMAGE_DIR, LABEL_DIR, train_files, transform=aug)
    for aug in augment_transforms
]

# 合并五倍数据
train_dataset = ConcatDataset(augmented_datasets)

# 验证集不需要数据增强
val_transform = transforms.ToTensor()
val_dataset = SegDataset(IMAGE_DIR, LABEL_DIR, val_files, transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# # 数据集与加载器
# train_dataset = SegDataset(IMAGE_DIR, LABEL_DIR, train_files, transform)
# val_dataset = SegDataset(IMAGE_DIR, LABEL_DIR, val_files, transform)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1)

# 模型与优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPlus(in_channels=3, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 指标记录
train_losses, train_accs = [], []
val_losses, val_accs, val_ious, val_dices = [], [], [], []

# 计算准确率、IoU、Dice
def calc_metrics(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1)
    target = target

    acc = (pred == target).sum().item() / target.numel()

    iou_list = []
    dice_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        iou = intersection / (union + 1e-8)

        dice = 2 * intersection / (pred_cls.sum().item() + target_cls.sum().item() + 1e-8)

        iou_list.append(iou)
        dice_list.append(dice)

    mean_iou = sum(iou_list) / num_classes
    mean_dice = sum(dice_list) / num_classes
    return acc, mean_iou, mean_dice

# 颜色映射
color_dict = [
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128)
]

# 训练与验证
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total_pixels = 0, 0, 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        loss = criterion(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        total_correct += (pred == mask).sum().item()
        total_pixels += mask.numel()

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_pixels
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss, val_correct, val_pixels = 0, 0, 0
    iou_sum, dice_sum = 0, 0
    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            val_loss += loss.item()

            acc, iou, dice = calc_metrics(output, mask, NUM_CLASSES)
            val_correct += acc * mask.numel()
            val_pixels += mask.numel()
            iou_sum += iou
            dice_sum += dice

            # 可视化保存预测
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()
            img_np = TF.to_pil_image(img[0].cpu())
            pred_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

            for cls in range(NUM_CLASSES):
                pred_rgb[pred == cls] = color_dict[cls]

            pred_img = Image.fromarray(pred_rgb)
            blended = Image.blend(img_np.convert('RGB'), pred_img, alpha=0.7)

            # 保存为与原图相同的文件名
            file_name = os.path.basename(val_files[idx].strip())
            save_path = os.path.join(PRED_DIR, file_name)
            blended.save(save_path)

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_pixels)
    val_ious.append(iou_sum / len(val_loader))
    val_dices.append(dice_sum / len(val_loader))

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"IOU: {val_ious[-1]:.4f} | Dice: {val_dices[-1]:.4f}")

# 保存曲线图
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(SAVE_DIR, 'loss_curve*.png'))

plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig(os.path.join(SAVE_DIR, 'accuracy_curve*.png'))

plt.figure()
plt.plot(val_ious, label='Val IOU')
plt.plot(val_dices, label='Val Dice')
plt.legend()
plt.title('IOU & Dice Curve')
plt.savefig(os.path.join(SAVE_DIR, 'iou_dice_curve*.png'))

# 保存模型参数
model_path = os.path.join(SAVE_DIR, 'unet*_model.pth')
torch.save(model.state_dict(), model_path)
print(f"模型已保存至: {model_path}")