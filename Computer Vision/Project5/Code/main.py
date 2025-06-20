import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from dataloader import get_dataloader  
from torchvision import transforms
from PIL import Image
from model import UNet
from tqdm import tqdm
import json
import time
from sklearn.metrics import confusion_matrix

# 创建保存目录
save_dir = './Result'
os.makedirs(save_dir, exist_ok=True)

# PASCAL VOC 数据集的颜色映射（RGB格式）
PALETTE = [
    [0, 0, 0],        # 背景
    [128, 0, 0],      # 飞机
    [0, 128, 0],      # 汽车
    [128, 128, 0],    # 汽船
    [0, 0, 128],      # 鸟
    [128, 0, 128],    # 猫
    [0, 128, 128],    # 奶牛
    [128, 128, 128],  # 狗
    [64, 0, 0],       # 马
    [192, 0, 0],      # 羊
    [64, 128, 0],     # 牛
    [192, 128, 0],    # 飞行器
    [64, 0, 128],     # 车辆
    [192, 0, 128],    # 卡车
    [64, 128, 128],   # 动物
    [192, 128, 128],  # 建筑
    [0, 64, 0],       # 运输工具
    [128, 64, 0],     # 水果
    [0, 64, 128],     # 工业
    [128, 64, 128]    # 其他
]

# 计算指标的函数
def calculate_metrics(outputs, targets, num_classes=21):
    # 将输出转换为预测类别
    _, predicted = torch.max(outputs, 1)
    
    # 展平预测和真实标签
    predicted_flat = predicted.view(-1)
    targets_flat = targets.view(-1)
    
    # 创建混淆矩阵
    conf_matrix = confusion_matrix(
        targets_flat.cpu().numpy(), 
        predicted_flat.cpu().numpy(),
        labels=np.arange(num_classes))
    
    # 计算各类别IoU
    iou_per_class = []
    for i in range(num_classes):
        true_positive = conf_matrix[i, i]
        false_positive = conf_matrix[:, i].sum() - true_positive
        false_negative = conf_matrix[i, :].sum() - true_positive
        
        # 避免除以零
        if true_positive + false_positive + false_negative == 0:
            iou_per_class.append(0.0)
        else:
            iou = true_positive / (true_positive + false_positive + false_negative)
            iou_per_class.append(iou)
    
    # 计算指标
    pixel_accuracy = np.trace(conf_matrix) / conf_matrix.sum()
    mean_accuracy = np.nanmean(np.diag(conf_matrix) / conf_matrix.sum(axis=1))
    mean_iou = np.nanmean(iou_per_class)
    
    # 计算频率加权IoU
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    fw_iou = (freq * np.array(iou_per_class)).sum()
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_accuracy': mean_accuracy,
        'mean_iou': mean_iou,
        'fw_iou': fw_iou,
        'iou_per_class': iou_per_class
    }

# 绘制训练曲线并保存
def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='mean_iou'):
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    
    # 指标曲线
    plt.subplot(2, 1, 2)
    plt.plot([m[metric_name] for m in train_metrics], label=f'Train {metric_name.upper()}')
    plt.plot([m[metric_name] for m in val_metrics], label=f'Validation {metric_name.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
    plt.legend()
    plt.title(f'Training and Validation {metric_name.upper()} Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{metric_name}.png'))
    plt.close()

# 保存指标到JSON文件
def save_metrics_to_json(train_metrics, val_metrics, filename='metrics.json'):
    metrics_data = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(metrics_data, f, indent=4)

def save_segmentation_example(model, val_loader, device, epoch=None):
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(val_loader))
        images, targets = images.to(device), targets.to(device)

        # 获取模型输出
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 转换为 numpy 数组，便于显示
        # 反标准化图像 - 修复维度问题
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        
        # 只处理第一个图像
        img_tensor = images[0].unsqueeze(0)  # 添加批次维度
        original_image = img_tensor * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        original_image = original_image.squeeze(0)  # 移除批次维度
        original_image = original_image.permute(1, 2, 0).cpu().numpy()
        
        ground_truth = targets[0].cpu().numpy()
        predicted_mask = predicted[0].cpu().numpy()

        # 使用颜色映射处理Ground Truth
        ground_truth_colored = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        for c in range(len(PALETTE)):
            ground_truth_colored[ground_truth == c] = PALETTE[c]

        # 使用颜色映射处理预测结果
        predicted_colored = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
        for c in range(len(PALETTE)):
            predicted_colored[predicted_mask == c] = PALETTE[c]

        # 处理显示
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 显示原始图像
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # 显示Ground Truth
        axs[1].imshow(ground_truth_colored)
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        # 显示预测的分割
        axs[2].imshow(predicted_colored)
        axs[2].set_title("Predicted Segmentation")
        axs[2].axis('off')

        # 保存图像
        if epoch is not None:
            plt.savefig(os.path.join(save_dir, f'segmentation_example_epoch_{epoch+1}.png'))
        else:
            plt.savefig(os.path.join(save_dir, 'segmentation_example_final.png'))
        plt.close()

# 训练过程（添加进度条和指标计算）
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    # 初始化指标计算
    all_outputs = []
    all_targets = []
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc='Training', leave=False)
    
    for batch_idx, (images, targets) in progress_bar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 收集预测结果用于指标计算
        all_outputs.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # 更新进度条描述
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
    
    # 计算训练指标
    outputs_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    train_metrics = calculate_metrics(outputs_tensor, targets_tensor)
    
    return running_loss / total_batches, train_metrics

# 验证过程（添加进度条和指标计算）
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_batches = len(val_loader)
    
    # 初始化指标计算
    all_outputs = []
    all_targets = []
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(enumerate(val_loader), total=total_batches, desc='Validation', leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in progress_bar:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # 收集预测结果用于指标计算
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            # 更新进度条描述
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')
    
    # 计算验证指标
    outputs_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    val_metrics = calculate_metrics(outputs_tensor, targets_tensor)
    
    return running_loss / total_batches, val_metrics

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 获取训练和验证数据加载器
    train_loader, val_loader = get_dataloader(batch_size=8)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # 检查一个批次的数据尺寸
    sample_images, sample_targets = next(iter(train_loader))
    print(f"Image batch shape: {sample_images.shape}")  
    print(f"Target batch shape: {sample_targets.shape}") 
    
    # 初始化模型、损失函数和优化器
    model = UNet(in_channels=3, out_channels=21).to(device)  # 21个类别
    criterion = nn.CrossEntropyLoss()  # 语义分割任务的损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 20
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    # 记录开始时间
    start_time = time.time()

    # 训练循环（外层进度条）
    epoch_progress = tqdm(range(num_epochs), desc='Epochs', total=num_epochs)
    
    for epoch in epoch_progress:
        # 训练一个epoch
        train_loss, train_metric = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        
        # 验证
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # 更新外层进度条描述
        epoch_progress.set_postfix(
            train_loss=f'{train_loss:.4f}', 
            val_loss=f'{val_loss:.4f}',
            mIoU=f'{val_metric["mean_iou"]:.4f}'
        )
        
        # 打印详细指标
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'  Train Pixel Acc: {train_metric["pixel_accuracy"]:.4f}, Val Pixel Acc: {val_metric["pixel_accuracy"]:.4f}')
        print(f'  Train Mean Acc: {train_metric["mean_accuracy"]:.4f}, Val Mean Acc: {val_metric["mean_accuracy"]:.4f}')
        print(f'  Train mIoU: {train_metric["mean_iou"]:.4f}, Val mIoU: {val_metric["mean_iou"]:.4f}')
        print(f'  Train FW IoU: {train_metric["fw_iou"]:.4f}, Val FW IoU: {val_metric["fw_iou"]:.4f}')
        
        # 每5个epoch保存一次示例和中间模型
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_segmentation_example(model, val_loader, device, epoch)
            print(f"Saved segmentation example at epoch {epoch+1}")
            
            # 保存中间模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, f'unet_model_epoch_{epoch+1}.pth'))
            print(f"Saved model checkpoint at epoch {epoch+1}")

    # 计算总训练时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # 绘制并保存训练曲线
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='mean_iou')
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='pixel_accuracy')
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='mean_accuracy')
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='fw_iou')
    
    # 保存最终分割示例图像
    save_segmentation_example(model, val_loader, device)
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }, os.path.join(save_dir, 'unet_model_final.pth'))
    print(f"Final model saved to {os.path.join(save_dir, 'unet_model_final.pth')}")
    
    # 保存指标到JSON文件
    save_metrics_to_json(train_metrics, val_metrics)
    
    # 打印最终指标
    final_val_metrics = val_metrics[-1]
    print("\nFinal Validation Metrics:")
    print(f"  Pixel Accuracy: {final_val_metrics['pixel_accuracy']:.4f}")
    print(f"  Mean Accuracy: {final_val_metrics['mean_accuracy']:.4f}")
    print(f"  Mean IoU: {final_val_metrics['mean_iou']:.4f}")
    print(f"  Frequency Weighted IoU: {final_val_metrics['fw_iou']:.4f}")
    
    # 打印各类别IoU
    print("\nPer-Class IoU:")
    for i, iou in enumerate(final_val_metrics['iou_per_class']):
        print(f"  Class {i}: {iou:.4f}")


if __name__ == '__main__':
    main()