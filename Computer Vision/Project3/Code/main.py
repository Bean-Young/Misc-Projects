import torch
import argparse
from model import SimpleCNN
from data_loader import get_data_loaders
from trainer import Trainer
from utils import get_device, plot_confusion_matrix, calculate_class_accuracy

def main(args):

    # 获取设备
    device = get_device()
    
    # 加载数据
    print("加载CIFAR10数据集...")
    train_loader, test_loader, classes = get_data_loaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    print("数据加载完成")
    
    # 创建模型
    print("创建模型...")
    model = SimpleCNN().to(device)
    print(model)
    
    # 创建训练器
    trainer = Trainer(model, device, train_loader, test_loader, classes)
    
    # 训练模型
    if not args.eval_only:
        print("开始训练模型...")
        history = trainer.train(epochs=args.epochs)
        
        # 绘制训练历史
        trainer.plot_history(history)
        
        # 保存模型
        trainer.save_model(args.model_path)
    else:
        # 加载已训练的模型
        print("加载已训练的模型...")
        trainer.load_model(args.model_path)
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    test_loss, test_accuracy = trainer.test()
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")
    
    # 可视化一些预测结果
    trainer.visualize_predictions(num_samples=8)
    
    # 计算混淆矩阵
    plot_confusion_matrix(model, test_loader, device, classes)
    
    # 计算每个类别的准确率
    calculate_class_accuracy(model, test_loader, device, classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10分类')
    parser.add_argument('--batch-size', type=int, default=128, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练的epoch数')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载器的工作进程数')
    parser.add_argument('--model-path', type=str, default='./Result/cifar10_model.pth', help='模型保存路径')
    parser.add_argument('--eval-only', action='store_true', help='仅评估，不训练')
    
    args = parser.parse_args()
    main(args)
