import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# 颜色映射（根据你的需求调整颜色）
color_dict = [
    (128, 0, 0),      # 类别 0
    (0, 128, 0),      # 类别 1
    (128, 128, 0),    # 类别 2
    (0, 0, 128)       # 类别 3
]

# 路径
IMAGE_DIR = '/home/yyz/Unet-ML/data/0.7/image'
LABEL_DIR = '/home/yyz/Unet-ML/data/0.7/label'
GT_DIR = '/home/yyz/Unet-ML/data/0.7/gt'
TRAIN_LIST = '/home/yyz/Unet-ML/data/0.7/train.txt'
VAL_LIST = '/home/yyz/Unet-ML/data/0.7/val.txt'
os.makedirs(GT_DIR, exist_ok=True)

def generate_gt_image(mask, num_classes=4):
    # 创建一个空的 RGB 图像
    gt_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for cls in range(num_classes):
        # 选择该类别的 mask 部分
        pred_cls = (mask == cls)
        gt_rgb[pred_cls] = color_dict[cls]  # 将对应区域赋予类别颜色

    return gt_rgb

def overlay_gt_on_image(image_path, mask_path, save_path, num_classes=4):
    # 加载原始图像和标签图像
    image = Image.open(image_path).convert('RGB')
    mask = np.array(Image.open(mask_path))  # 加载标签图像并转换为 numpy 数组
    
    # 生成 GT 图像
    gt_image = generate_gt_image(mask, num_classes)
    gt_image = Image.fromarray(gt_image)
    
    # 叠加原图和 GT 图像（使用 alpha 值控制叠加强度）
    blended_image = Image.blend(image, gt_image.convert('RGB'), alpha=0.7)
    
    # 保存叠加后的图像
    blended_image.save(save_path)

# 获取训练和验证列表中的文件名
with open(TRAIN_LIST, 'r') as f:
    train_files = f.readlines()

with open(VAL_LIST, 'r') as f:
    val_files = f.readlines()

# 遍历训练和验证图像和标签，进行处理
for file_list, split in zip([train_files, val_files], ['train', 'val']):
    for file_name in file_list:
        file_name = file_name.strip()  # 去掉多余的空格或换行符
        
        # 构建图像和标签路径
        image_path = os.path.join(IMAGE_DIR, file_name)
        label_path = os.path.join(LABEL_DIR, file_name)
        
        # 提取文件名前缀（去掉扩展名）
        file_prefix = os.path.splitext(file_name)[0]
        
        # 设置保存路径（与原图像名称相同）
        save_path = os.path.join(GT_DIR, f'{file_prefix}.png')
        
        # 叠加标签并保存
        overlay_gt_on_image(image_path, label_path, save_path)
        print(f"Processed {split} {file_prefix}: {save_path}")