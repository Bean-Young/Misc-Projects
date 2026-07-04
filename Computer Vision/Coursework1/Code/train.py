"""
Training script for ResNet on CIFAR-10
Computer Vision Course Design 1
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from resnet import ResNet18, ResNet34
from data_loader import get_cifar10_data_loaders, extract_cifar10_data


class Trainer:
    """Trainer class for ResNet on CIFAR-10"""

    def __init__(self, model, train_loader, test_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 150], gamma=0.1
        )

        # TensorBoard writer
        self.writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))

        # Best accuracy tracking
        self.best_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

            # Log to TensorBoard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, step)

        # Calculate average loss and accuracy
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def test(self, epoch):
        """Test the model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        # Log to TensorBoard
        self.writer.add_scalar('Test/Loss', avg_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', accuracy, epoch)

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'args': self.args
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {accuracy:.2f}%")

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.args.model}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Batch size: {self.args.batch_size}")

        start_time = time.time()

        for epoch in range(self.args.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Test
            test_loss, test_acc = self.test(epoch)

            # Update learning rate
            self.scheduler.step()

            # Print epoch results
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)

            # Save checkpoint
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc

            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, test_acc, is_best)

        # Training completed
        total_time = time.time() - start_time
        print(f'Training completed in {total_time/3600:.2f} hours')
        print(f'Best test accuracy: {self.best_acc:.2f}%')

        # Close TensorBoard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='ResNet CIFAR-10 Training')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--data_dir', type=str, default='/data/public/yyz/cv/data/cifar-10-batches-py',
                       help='CIFAR-10 data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/public/yyz/cv/models',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='/data/public/yyz/cv/logs',
                       help='Log directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading CIFAR-10 data...")
    train_loader, test_loader = get_cifar10_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # Create model
    if args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'resnet34':
        model = ResNet34()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model created: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer and start training
    trainer = Trainer(model, train_loader, test_loader, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
