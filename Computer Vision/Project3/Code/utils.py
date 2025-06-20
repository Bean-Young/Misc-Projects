import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置全局中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号乱码
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')

    return device

def plot_confusion_matrix(model, test_loader, device, classes):

    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('./Result/confusion_matrix.png')
    plt.close()  # 关闭图像而不是显示

def calculate_class_accuracy(model, test_loader, device, classes):
    # 确保结果目录存在
    os.makedirs('./Result', exist_ok=True)
    
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            c = (preds == targets).squeeze()
            
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算并打印每个类别的准确率
    print("\n每个类别的准确率:")
    class_accuracies = []
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {accuracy:.2f}%')
        class_accuracies.append(accuracy)
    
    # 绘制每个类别的准确率
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_accuracies)
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title('每个类别的准确率')
    plt.ylim([0, 100])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./Result/class_accuracy.png')
    plt.close()  # 关闭图像而不是显示
    
    return class_accuracies
