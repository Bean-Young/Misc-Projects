import os
import random
import shutil

# 参数配置
train_percentage = 0.7

# 原始数据路径
image_dir = '/home/yyz/Unet-ML/data/image/train_data'
label_dir = '/home/yyz/Unet-ML/data/label/train_label'

# 新目录配置
base_save_dir = '/home/yyz/Unet-ML/data/1.0'
save_image_dir = os.path.join(base_save_dir, 'image')
save_label_dir = os.path.join(base_save_dir, 'label')
os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)

# 读取图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
image_files.sort()

# 数据划分
total_num = len(image_files)
train_num = int(total_num * train_percentage)
index_list = list(range(total_num))
train_indices = random.sample(index_list, train_num)

# 保存路径
train_txt_path = os.path.join(base_save_dir, 'train.txt')
val_txt_path = os.path.join(base_save_dir, 'val.txt')

# 写入文件 & 拷贝图像标签
with open(train_txt_path, 'w') as f_train, open(val_txt_path, 'w') as f_val:
    for idx in index_list:
        filename = image_files[idx]
        line = filename + '\n'

        src_image = os.path.join(image_dir, filename)
        src_label = os.path.join(label_dir, filename)
        dst_image = os.path.join(save_image_dir, filename)
        dst_label = os.path.join(save_label_dir, filename)

        # 拷贝文件
        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)

        # 写入对应 txt
        if idx in train_indices:
            f_train.write(line)
        else:
            f_val.write(line)

print(f"✅ Train/Val split and copy done.")
print(f"Total: {total_num} | Train: {train_num} | Val: {total_num - train_num}")
print(f"Train file: {train_txt_path}")
print(f"Val file:   {val_txt_path}")
print(f"Images and labels copied to:")
print(f" - {save_image_dir}")
print(f" - {save_label_dir}")