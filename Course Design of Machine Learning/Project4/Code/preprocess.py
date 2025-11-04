import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

# 参数配置
percentage_threshold = 0.7
train_percentage = 0.7

# 路径配置
image_dir = '/home/yyz/Unet-ML/data/image/train_data'
label_dir = '/home/yyz/Unet-ML/data/label/train_label'
save_image_dir = '/home/yyz/Unet-ML/data/0.7/image'
save_label_dir = '/home/yyz/Unet-ML/data/0.7/label'
os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)

# 筛选符合条件的图像
qualified_filenames = []
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

for filename in tqdm(image_files, desc="Checking label distribution"):
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename)

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f"Warning: Label not found or unreadable: {label_path}")
        continue

    total_pixels = label.size
    unique, counts = np.unique(label, return_counts=True)
    stats = dict(zip(unique, counts))

    # 检查四类中每类是否都低于阈值
    if all(stats.get(i, 0) / total_pixels < percentage_threshold for i in [0, 1, 2, 3]):
        qualified_filenames.append(filename)
        # 复制图像与标签到 0.7 文件夹
        shutil.copy(image_path, os.path.join(save_image_dir, filename))
        shutil.copy(label_path, os.path.join(save_label_dir, filename))

print(f"Qualified samples: {len(qualified_filenames)}")

# 数据划分
total_num = len(qualified_filenames)
train_num = int(total_num * train_percentage)
index_list = list(range(total_num))
train_indices = random.sample(index_list, train_num)

train_txt_path = '/home/yyz/Unet-ML/data/0.7/train.txt'
val_txt_path = '/home/yyz/Unet-ML/data/0.7/val.txt'

with open(train_txt_path, 'w') as f_train, open(val_txt_path, 'w') as f_val:
    for idx in index_list:
        filename = qualified_filenames[idx]
        line = filename + '\n'
        if idx in train_indices:
            f_train.write(line)
        else:
            f_val.write(line)

print("Train/Val split done.")
print(f"Train size: {train_num}")
print(f"Val size: {total_num - train_num}")