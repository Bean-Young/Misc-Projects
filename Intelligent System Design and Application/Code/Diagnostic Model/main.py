import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,6"
import sys
import argparse
import time
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
from visualizer import get_local
get_local.activate()
from dataset.data_loader import load_dataset
from dataset.data_visual import visual_data

from tools.utils import Multi_Accuracy
from tools.loss import CustomLoss
from tools.init import Initialization
from tools.adapt import FC_class
from tools.visualizer import *

from model.c3d import C3D
from model.RViT import RViT
from model.resnet import  ResNet3D
from model.vgg import VGG3D
from model.slowfast import SlowFast
from model.vivit import ViViT
from timesformer.models.vit import TimeSformer
from model.usvit import US_RViT_Cov
from model.ussf import US_RViT
import sys
import collections.abc

import sys
import argparse
import time
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tensorboardX import SummaryWriter
from visualizer import get_local
get_local.activate()
from dataset.data_loader import load_dataset
from dataset.data_visual import visual_data

from tools.utils import Multi_Accuracy
from tools.loss import CustomLoss
from tools.init import Initialization
from tools.adapt import FC_class
from tools.visualizer import *


sys.modules['torch._six'] = sys.modules[__name__]
container_abcs = collections.abc


frozen_list =['embedding.weight', 'embedding.bias']

parser = argparse.ArgumentParser()

# Dataset Config
parser.add_argument('--data_dir', type=str, default='/media/Storage4/yyz_data/usrvit',
                    help='the path of data')
parser.add_argument('--label_dir', type=str, default='/media/Storage4/yyz_data/usrvit',
                    help='the path of label')
parser.add_argument('--model_path', type=str, default=None,    #'/home/yyz/result/writer/model_RTrans_68.pth' #'/media/Storage4/yyz_code/MDPI-US/res_yyz/vgg/model_RTrans_30.pth'
                    help='the path of pretrained model')#'/home/yyz/RViT-main/model/final.pth'
parser.add_argument('--model_save_dir', type=str, default='/media/Storage4/yyz_code/MDPI-US/res_yyz/ourcwofa/',
                    help='the path of the trained model')
parser.add_argument('--writer_save_dir', type=str, default='/media/Storage4/yyz_code/MDPI-US/res_yyz/ourc/',
                    help='the path of the writer')

# Model Config
parser.add_argument('--layer_num', type=int, default=2,
                    help='RViT layer numbers')
parser.add_argument('--class_num', type=int, default=5,
                    help='the number of action categories')
parser.add_argument('--sample_length', type=int, default=50,
                    help='the length of singal clip')
parser.add_argument('--image_size', type=tuple, default=(224,224),
                    help="The size of image size, format is (int, int)")
parser.add_argument('--crop_size', type=int, default=224,
                    help="The size of image cropping size, format is int")
parser.add_argument('--patch_size', type=tuple, default=(16,16),
                    help="The size of each patch")
parser.add_argument('--num_heads', type=int, default=8,
                    help="The number of heads in Multi-Head attention")
parser.add_argument('--embed_type', type=str, default='conv',
                    help="The embedding type we use, select 'conv' or 'net'")
parser.add_argument('--attn_type', type=str, default='linear',
                    help="The attention type we use, select 'linear', 'softmax' or 'reattn'")
parser.add_argument('--dropout', type=float, default=0.0,
                    help="The dropout rate RViT")

# Training Parameters
parser.add_argument('--train', action='store_true', default=True,
                    help='Use the training mode or validation mode')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Do the attention map visualize')
parser.add_argument('--distributed', type=bool, default=False,
                    help='Using distributed training')
parser.add_argument('--enable_GPUs_id', type=list, default=[0,1,2,3],
                    help = 'gpu devices ids')
parser.add_argument('--local_rank', type=int,default=0,
                    help='using which GPU(s)')
parser.add_argument('--layer_frozen', action='store_true', default=False,
                    help = 'setting the type of optimizer',)
parser.add_argument('--batch_size', type=int, default=2,
                    help='size for each minibatch')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--optimizer', type=str, help = 'setting the type of optimizer',
                    default='AdamW')
parser.add_argument('--loss_type', type=list, default=['LSCE'],
                    help='the type of criterion')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--warm_up', type=int, default=1,
                    help='warm up epoch')
parser.add_argument('--step_size', type=int, default=10,
                    help='initial step_size')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='initial betas param')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay rate')
parser.add_argument('--clip_grad', action='store_true', default=False,
                    help='Using the clip gradient normlization')
parser.add_argument('--seed', type=int, default=8415,
                    help='seed for random initialisation')
parser.add_argument('--num_workers', type=int, help = 'setting the workers number',
                    default=4)
parser.add_argument('--model_depth', type=int, default=18,
                    help='ResNet3D模型深度[10, 18, 34, 50, 101, 152, 200]')
parser.add_argument('--n_input_channels', type=int, default=50,
                    help='输入通道数') #for resnet  is 50 while vgg is 3.

args = parser.parse_args()


class Demo():
    def __init__(self, debug=False):
        self.best_metrics = {
            'top1': 0,
            'top3': 0,
            'top5': 0,
            'acc': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'loss': float('inf'),
        }
        self.best_class_auc = {class_name: 0 for class_name in ['Cov', 'Benign', 'Malignant', 'Gall', 'Pneu']}
        torch.backends.cudnn.benchmark = True
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
        self.init = Initialization()
        self.device, self.local_rank = self.init.init_params(args.enable_GPUs_id, args.distributed)
        print("self.local_rank=",self.local_rank,"GPU_ids=",args.enable_GPUs_id[0])
        # self.model = US_RViT(image_size = args.image_size[0],
        #                   patch_size = args.patch_size[0],
        #                   num_classes = args.class_num,
        #                   depth = args.layer_num,
        #                   length = args.sample_length,
        #                   heads = args.num_heads,
        #                   mlp_dim = args.patch_size[0] ** 2 * 3 * 4,
        #                   dropout = args.dropout,
        #                   batch = args.batch_size)#✅

        # self.model = ResNet3D(
        #     num_classes=args.class_num,
        #     model_depth=args.model_depth,
        #     in_channels=args.n_input_channels
        # )#✅

        # self.model = VGG3D(
        #             num_classes=args.class_num,
        #             in_channels=args.n_input_channels
        #         ) #✅

        # self.model = C3D() #✅

        # self.model = SlowFast(num_classes=args.class_num)
        # self.alpha = 4


        # self.model = ViViT(
        #     num_classes=args.class_num,
        #     num_frames=args.sample_length
        # )
        # 用 TimeSformer 替换 ViViT
        # self.model = TimeSformer(
        #     img_size=args.image_size[0],      # 图像的大小（根据需要修改）
        #     patch_size=args.patch_size[0],    # Patch的大小
        #     num_classes=args.class_num,       # 类别数
        #     num_frames=args.sample_length,    # 每个视频的帧数
        #     attention_type='divided_space_time',  # 注意力类型，可以根据需要更改
        # )

        self.model = US_RViT_Cov()
        #         # 优化器设置

        # 1. 读取 checkpoint
        # checkpoint = torch.load(args.model_path, map_location=self.device)

        # # 2. 处理 key 的 `module.` 前缀
        # new_state_dict = {}
        # for key, value in checkpoint["net"].items():
        #     new_key = key.replace("module.", "")  # 去掉 "module." 前缀
        #     new_state_dict[new_key] = value

        # # 3. 加载处理后的 state_dict
        # self.model.load_state_dict(new_state_dict)

        if args.model_path :
            checkpoint = torch.load(args.model_path, map_location=self.device)
            new_state_dict = {}
            for key, value in checkpoint["net"].items():
                new_key = key.replace("module.", "")  # 如果有 "module." 前缀，去掉它
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)

        self.model = self.init.to_GPU(self.model, self.device, self.local_rank)
        #print(self.model)


        if args.layer_frozen:
            self.model = self.init.frozen_layer(self.model, frozen_list)

        self.optimizer, self.scheduler = self.init.optimizer_init(self.model,
                                                                  args.optimizer,
                                                                  args.lr,
                                                                  args.step_size,
                                                                  args.betas,
                                                                  args.weight_decay)

        #self.criterion = CustomLoss()
        self.criterion = self.init.init_criterion(self.device,args.loss_type)
        self.model = self.init.use_multi_GPUs(self.model, self.local_rank, args.enable_GPUs_id, args.distributed)

        param = {'w':args.image_size[0],
                 'h':args.image_size[1],
                 'crop_size':args.crop_size,
                 'sample_length':args.sample_length,
                 'video_dirs':args.data_dir,
                 'label_dir':args.label_dir}

        if args.train:
            self.train_dataset = load_dataset(args=param, split='train')

            if args.distributed:
                self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                               rank=self.local_rank,
                                                                               shuffle=True,)
            else:
                self.sampler = None

            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           shuffle=(self.sampler is None),
                                           drop_last=True,
                                           sampler=self.sampler)

        if args.visualize:
            self.val_dataset = visual_data(args=param, split='validation')
        else:
            self.val_dataset = load_dataset(args=param, split='validation')

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=1 if args.visualize else args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=True,
                                     drop_last=True) if self.local_rank == args.enable_GPUs_id[0] else None

        self.summary, self.print_info, self.count, self.val_count = {}, {}, 0, 0
        log_dir = os.path.join(args.writer_save_dir, 'loss')
        os.makedirs(log_dir, exist_ok=True)  # 👈 加这个，避免 FileExistsError
        self.writer = SummaryWriter(log_dir)
        # self.writer = SummaryWriter(os.path.join(args.writer_save_dir, 'loss'))

        self.h_0 = torch.autograd.Variable(torch.zeros(args.layer_num,
                                                       1 if args.visualize else args.batch_size,
                                                       (args.image_size[0]//args.patch_size[0])**2,
                                                       args.patch_size[0] ** 2 * 3,
                                                       requires_grad=True)).to(self.device)

    def train(self):
        for self.epoch in range(args.num_epochs):

            if self.epoch < args.warm_up:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = args.lr * (10 ** (self.epoch-args.warm_up+1))

            print("current learning rate : ", param_group['lr'])
            print('Start Epoch: {}'.format(self.epoch))

            self.model.train()
            for self.step, (inputs, labels,index) in enumerate(tqdm(self.train_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # result = self.model(inputs,self.h_0)
                # fast_pathway = inputs.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                # slow_pathway = fast_pathway[:, :, ::self.alpha, :, :]  # 每 alpha 帧采样一帧
                # result = self.model([slow_pathway, fast_pathway])
                # result = self.model(inputs.permute(0, 2, 1, 3, 4))  # 变成 (batch, channels, depth, height, width) #vgg vivit
                result =self.model(inputs) #Resnet
                # result = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                #loss = self.criterion(result, labels)
                loss = self.criterion[0](result, labels)

                top1,top3, top5 = Multi_Accuracy(result.data, labels, topk=(1,3,5))

                if self.local_rank == args.enable_GPUs_id[0]:
                    self.count += 1
                    self.add_summary(self.writer, 'train/loss', loss.item())
                    self.add_summary(self.writer, 'train/top1', top1.item())
                    self.add_summary(self.writer, 'train/top3', top3.item())
                    self.add_summary(self.writer, 'train/top5', top5.item())

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        tag = tag.replace('module', '')
                        self.add_summary(self.writer, tag, value.data.cpu().numpy(), sum_type='histogram')

                self.optimizer.zero_grad()

                loss.backward(retain_graph=True)
                if args.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

                self.optimizer.step()

            self.scheduler.step()

            if self.local_rank == args.enable_GPUs_id[0]:
                print('------Training Result------\n \
                       Top-1 accuracy: {top1_acc:.2f}%, \
                       Top-3 accuracy: {top3_acc:.2f}%, \
                       Top-5 accuracy: {top5_acc:.2f}% \
                       Loss value: {loss:.2f}'.\
                       format(top1_acc=self.print_info['train/top1'],
                              top3_acc=self.print_info['train/top3'],
                              top5_acc=self.print_info['train/top5'],
                              loss=self.print_info['train/loss'],)
                              )
                print('End Training Epoch: {}'.format(self.epoch))

            self.model.eval()
            self.validation()
            self.save()
            with open(os.path.join(args.model_save_dir, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'metrics': self.best_metrics,
                    'class_auc': self.best_class_auc
                }, f, indent=4)


    def validation(self):
        total_top1, total_top3, total_top5 = 0, 0, 0
        total_accuracy, total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0, 0

        if self.local_rank == args.enable_GPUs_id[0]:
            all_labels = []
            all_preds = []
            all_probs = []  # 用于计算 AUC
            all_indices = []  # 用于存储索引编号

            label_mapping = {
                'Cov': 0,
                'Benign': 1,
                'Malignant': 2,
                'Gall': 3,
                'Pneu': 4,
            }
            inv_label_mapping = {v: k for k, v in label_mapping.items()}  # 反向映射

            with torch.no_grad():
                for self.step, (inputs, labels, indices) in enumerate(tqdm(self.val_loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # fast_pathway = inputs.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                    # slow_pathway = fast_pathway[:, :, ::self.alpha, :, :]  # 每 alpha 帧采样一帧
                    # result = self.model([slow_pathway, fast_pathway])
                    # # result = self.model(inputs.permute(0, 2, 1, 3, 4))
                    result =self.model(inputs)
                    # result = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                    # result = self.model(inputs, self.h_0)
                    loss = self.criterion[0](result, labels)

                    # 计算 Top-K Accuracy
                    top1, top3, top5 = Multi_Accuracy(result.data, labels, topk=(1,3,5))

                    # 记录 loss 和 accuracy
                    self.val_count += 1
                    self.add_summary(self.writer, 'val/loss', loss.item(), record_type='validation')
                    self.add_summary(self.writer, 'val/top1', top1.item(), record_type='validation')
                    self.add_summary(self.writer, 'val/top3', top3.item(), record_type='validation')
                    self.add_summary(self.writer, 'val/top5', top5.item(), record_type='validation')

                    total_top1 += top1.item()
                    total_top3 += top3.item()
                    total_top5 += top5.item()

                    probs = F.softmax(result, dim=1)
                    pred_labels = probs.argmax(dim=1)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(pred_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())  # 保存 softmax 概率
                    all_indices.extend(indices.cpu().numpy())  # 存储索引

            # 计算分类性能
            total_accuracy = accuracy_score(all_labels, all_preds)
            total_precision = precision_score(all_labels, all_preds, average='macro')
            total_recall = recall_score(all_labels, all_preds, average='macro')
            total_f1 = f1_score(all_labels, all_preds, average='macro')

            # 计算每个类别的 AUC
            class_aucs = {}
            all_labels_np = np.array(all_labels)
            all_probs_np = np.array(all_probs)
            for i in range(len(label_mapping)):
                try:
                    class_auc = roc_auc_score((all_labels_np == i).astype(int), all_probs_np[:, i])
                except ValueError:
                    class_auc = float('nan')  # 该类别在标签中可能没有出现
                class_name = inv_label_mapping[i]
                class_aucs[class_name] = class_auc

            # 创建并保存结果 CSV
            df = pd.DataFrame({
                "Sample_ID": all_indices,
                "True_Label": [inv_label_mapping[label] for label in all_labels],
                "Pred_Label": [inv_label_mapping[label] for label in all_preds],
            })
            df = df.sort_values(by="Sample_ID")
            df.to_csv("validation_results.csv", index=False)
            print("Validation results saved to validation_results.csv")

            # 打印整体指标
            print('------Validation Summary------\n \
                Top-1 accuracy: {top1_acc:.2f}%, \
                Top-3 accuracy: {top3_acc:.2f}%, \
                Top-5 accuracy: {top5_acc:.2f}%, \
                Accuracy :{acc:.2f}%, \
                Precision: {precision:.2f}%, \
                Recall: {recall:.2f}%, \
                F1-Score: {f1:.2f}, \
                Loss value: {loss:.2f}'.format(
                    top1_acc=total_top1/self.step,
                    top3_acc=total_top3/self.step,
                    top5_acc=total_top5/self.step,
                    acc=total_accuracy * 100,
                    precision=total_precision * 100,
                    recall=total_recall * 100,
                    f1=total_f1,
                    loss=self.print_info['val/loss'])
            )

            # 打印每个类别的 AUC
            print("------Per-class AUC------")
            for class_name, auc_val in class_aucs.items():
                print(f"{class_name}: AUC = {auc_val:.4f}")

            self.update_best_metrics('top1', total_top1 / self.step)
            self.update_best_metrics('top3', total_top3 / self.step)
            self.update_best_metrics('top5', total_top5 / self.step)
            self.update_best_metrics('acc', total_accuracy)
            self.update_best_metrics('precision', total_precision)
            self.update_best_metrics('recall', total_recall)
            self.update_best_metrics('f1', total_f1)
            self.update_best_metrics('loss', self.print_info['val/loss'])

            self.update_best_auc(class_aucs)

    def update_best_metrics(self, metric_name, value):
        if metric_name == 'loss':
            if value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                print(f"[Best] Updated best {metric_name}: {value:.4f}")
        else:
            if value > self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                print(f"[Best] Updated best {metric_name}: {value:.4f}")

    def update_best_auc(self, class_aucs):
        for class_name, auc_val in class_aucs.items():
            if auc_val > self.best_class_auc[class_name]:
                self.best_class_auc[class_name] = auc_val
                print(f"[Best] Updated best AUC for {class_name}: {auc_val:.4f}")
    # def validation(self):

    #     total_top1, total_top3, total_top5 = 0, 0, 0
    #     total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0

    #     if self.local_rank == args.enable_GPUs_id[0]:
    #         all_labels = []
    #         all_preds = []
    #         all_probs = []  # 用于计算 AUC

    #         with torch.no_grad():
    #             for self.step, (inputs, labels) in enumerate(tqdm(self.val_loader)):
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)
    #                 # fast_pathway = inputs.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    #                 # slow_pathway = fast_pathway[:, :, ::self.alpha, :, :]  # 每 alpha 帧采样一帧
    #                 # result = self.model([slow_pathway, fast_pathway])
    #                 # result = self.model(inputs.permute(0, 2, 1, 3, 4))
    #                 result =self.model(inputs)
    #                 # result = self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
    #                 loss = self.criterion[0](result, labels)

    #                 # 计算 Top-K Accuracy
    #                 top1, top3, top5 = Multi_Accuracy(result.data, labels, topk=(1,3,5))

    #                 # 记录 loss 和 accuracy
    #                 self.val_count += 1
    #                 self.add_summary(self.writer, 'val/loss', loss.item(), record_type='validation')
    #                 self.add_summary(self.writer, 'val/top1', top1.item(), record_type='validation')
    #                 self.add_summary(self.writer, 'val/top3', top3.item(), record_type='validation')
    #                 self.add_summary(self.writer, 'val/top5', top5.item(), record_type='validation')

    #                 total_top1 += top1.item()
    #                 total_top3 += top3.item()
    #                 total_top5 += top5.item()

    #                 # 计算 Precision / Recall / F1-Score
    #                 probs = F.softmax(result, dim=1)  # 转换成概率
    #                 pred_labels = probs.argmax(dim=1)  # 取最大概率类别

    #                 all_labels.extend(labels.cpu().numpy())
    #                 all_preds.extend(pred_labels.cpu().numpy())
    #                 all_probs.extend(probs.cpu().numpy())  # 保存 softmax 结果

    #         # 计算 Precision / Recall / F1
    #         total_precision = precision_score(all_labels, all_preds, average='macro')
    #         total_recall = recall_score(all_labels, all_preds, average='macro')
    #         total_f1 = f1_score(all_labels, all_preds, average='macro')

    #         # 计算 AUC（针对 5 分类任务）
    #         try:
    #             total_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    #         except:
    #             total_auc = 0  # 避免 AUC 计算失败

    #         print('------Validation Result------\n \
    #             Top-1 accuracy: {top1_acc:.2f}%, \
    #             Top-3 accuracy: {top3_acc:.2f}%, \
    #             Top-5 accuracy: {top5_acc:.2f}% \
    #             Precision: {precision:.2f}%, \
    #             Recall: {recall:.2f}%, \
    #             F1-Score: {f1:.2f}, \
    #             AUC: {auc:.2f}, \
    #             Loss value: {loss:.2f}'.\
    #             format(top1_acc=total_top1/self.step,
    #                 top3_acc=total_top3/self.step,
    #                 top5_acc=total_top5/self.step,
    #                 precision=total_precision * 100,
    #                 recall=total_recall * 100,
    #                 f1=total_f1,
    #                 auc=total_auc,
    #                 loss=self.print_info['val/loss']))
    #     else:
    #         pass

    def visualize(self):
        with torch.no_grad():
            for self.step, (inputs, labels,frame_list) in enumerate(tqdm(self.val_loader)):
                #inputs, labels, redundance_label = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                result = self.model(inputs.permute(0, 2, 1, 3, 4))
                # result= self.model(inputs.view(-1, 3, args.image_size[0], args.image_size[1]), self.h_0)
                cache = get_local.cache
                if self.step == 0:
                    print(cache.keys())
                    attention_maps = cache['MultiHeadDotProductAttention.linear_attn']
                    vid_len = len(frame_list)
                    for i in range(1,vid_len*2,2):
                        #visualize_cto_grid_with_cls(attention_maps[i][0,h,:,:], 105, np.array(frame_list[i//2][0]), num=i, head=h)
                        #visualize_grid_to_grid(attention_maps[i][0,h,1:,1:], 100, np.array(frame_list[i//2][0]), num=i, head=h)
                        for j in range(args.num_heads):
                            attention_map = np.mean(attention_maps[i][0,:,:,:],axis=0,keepdims=False)
                            visualize_grid_to_grid_with_cls(attention_map, 98, np.array(frame_list[i//2][0]),alpha=0.55, num=i, head=j)

                        visualize_heads(attention_maps[i],cols=4,num=i//2)


    # add summary
    def add_summary(self, writer, name, val, sum_type = 'scalar', record_type = 'train'):
        def writer_in(writer, name, val, sum_type, count):
            if sum_type == 'scalar':
                writer.add_scalar(name, self.summary[name]/5, count)
                self.print_info[name] = self.summary[name]/5
                self.summary[name] = 0
            elif sum_type == 'image':
                writer.add_image(name, val, count)
            elif sum_type == 'histogram':
                writer.add_histogram(name, val, count)

        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.count % 5 == 0 and record_type == 'train':
            writer_in(writer, name, val, sum_type, self.count)
        elif writer is not None and self.val_count % 5 == 0 and record_type == 'validation':
            writer_in(writer, name, val, sum_type, self.val_count)

    def save(self):
        torch.save({
            'net': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(args.model_save_dir, 'model_RTrans_{}.pth'.format(self.epoch)))


def main():
    process = Demo()
    if args.train:
        process.train()
    elif args.visualize:
        process.visualize()
    else:
        process.validation()

if __name__ == "__main__":
    main()