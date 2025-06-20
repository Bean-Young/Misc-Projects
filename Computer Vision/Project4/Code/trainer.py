import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import matplotlib

# 设置全局中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号乱码
import numpy as np
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, device, train_loader, test_loader, classes):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = classes
        
        # 创建结果目录
        os.makedirs('./Result', exist_ok=True)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5, verbose=True
        )
        
        # 训练历史记录
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(self.train_loader, desc='Training')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        train_loss = running_loss / len(self.train_loader)
        train_accuracy = 100. * correct / total
        
        return train_loss, train_accuracy
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100. * correct / total
        
        return test_loss, test_accuracy
    
    def train(self, epochs=50):
        print(f"开始训练，总共{epochs}个epochs...")
        start_time = time.time()
        
        best_acc = 0.0  # 记录最佳准确率
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # 在测试集上评估
            test_loss, test_accuracy = self.test()
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            # 学习率调度器
            self.scheduler.step(test_loss)
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
            print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")
            
            # 保存最佳模型
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                self.save_model('./Result/cifar100_resnet18_best.pth')
                print(f"保存最佳模型，准确率: {best_acc:.2f}%")
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f}秒")
        print(f"最佳测试准确率: {best_acc:.2f}%")
        
        # 返回训练历史记录
        history = {
            'train_loss': self.train_losses,
            'test_loss': self.test_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy': self.test_accuracies
        }
        
        return history
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['test_loss'], label='测试损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练和测试损失')
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='训练准确率')
        plt.plot(history['test_accuracy'], label='测试准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.title('训练和测试准确率')
        
        plt.tight_layout()
        plt.savefig('./Result/training_history.png')
        plt.close()  # 关闭图像而不是显示
        
    def visualize_predictions(self, num_samples=5):
        """
        可视化模型在测试集上的一些预测结果
        
        参数:
            num_samples: 要可视化的样本数量
        """
        self.model.eval()
        
        # 获取一批测试数据
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        
        # 选择前num_samples个样本
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # 预测
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        # 转换图像用于显示
        images = images.cpu().numpy()
        
        # 显示图像和预测结果
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i in range(num_samples):
            # 反归一化图像
            img = np.transpose(images[i], (1, 2, 0))
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"真实: {self.classes[labels[i]]}\n预测: {self.classes[predicted[i]]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('./Result/predictions.png')
        plt.close()  # 关闭图像而不是显示
        
    def save_model(self, path='./Result/cifar100_resnet18_model.pth'):
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")
        
    def load_model(self, path='./Result/cifar100_resnet18_model.pth'):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {path} 加载")
