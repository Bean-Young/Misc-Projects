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
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')

    return device

def plot_confusion_matrix(model, test_loader, device, classes, num_classes_to_show=10):

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
    
    # 计算完整的混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 选择前num_classes_to_show个类别进行可视化
    cm_subset = cm[:num_classes_to_show, :num_classes_to_show]
    classes_subset = classes[:num_classes_to_show]
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_subset, yticklabels=classes_subset)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (前{num_classes_to_show}个类别)')
    plt.tight_layout()
    plt.savefig('./Result/confusion_matrix.png')
    plt.close()  # 关闭图像而不是显示
    
    # 保存完整的混淆矩阵数据
    np.save('./Result/full_confusion_matrix.npy', cm)

def calculate_class_accuracy(model, test_loader, device, classes, top_n=20):
  
    model.eval()
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
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
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        class_accuracies.append(accuracy)
    
    # 创建类别准确率的数据框
    class_acc_data = [(classes[i], class_accuracies[i]) for i in range(num_classes)]
    
    # 按准确率排序
    class_acc_data.sort(key=lambda x: x[1], reverse=True)
    
    # 获取准确率最高和最低的类别
    top_classes = class_acc_data[:top_n]
    bottom_classes = class_acc_data[-top_n:]
    
    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    for class_name, accuracy in class_acc_data:
        print(f'{class_name}: {accuracy:.2f}%')
    
    # 绘制准确率最高的类别
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar([x[0] for x in top_classes], [x[1] for x in top_classes])
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title(f'准确率最高的{top_n}个类别')
    plt.ylim([0, 100])
    plt.xticks(rotation=90)
    
    # 绘制准确率最低的类别
    plt.subplot(1, 2, 2)
    plt.bar([x[0] for x in bottom_classes], [x[1] for x in bottom_classes])
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.title(f'准确率最低的{top_n}个类别')
    plt.ylim([0, 100])
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('./Result/class_accuracy.png')
    plt.close()  # 关闭图像而不是显示
    
    # 保存所有类别准确率数据
    np.save('./Result/class_accuracies.npy', np.array(class_accuracies))
    
    return class_accuracies

def calculate_top_k_accuracy(model, test_loader, device, k=5):

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 获取前k个预测结果
            _, pred = outputs.topk(k, 1, True, True)
            pred = pred.t()
            correct_k = pred.eq(targets.view(1, -1).expand_as(pred)).sum().item()
            
            correct += correct_k
            total += targets.size(0)
    
    top_k_accuracy = 100 * correct / total
    print(f"\nTop-{k}准确率: {top_k_accuracy:.2f}%")
    
    return top_k_accuracy
