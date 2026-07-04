import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 以添加进度条

# 指定 .npy 文件的目录路径
directory = '/media/Storage2/yyz/Pretreat'

# 收集所有 .npy 文件的路径
npy_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.npy'):
            npy_files.append(os.path.join(root, file))

# 使用 tqdm 添加进度条
for file_path in tqdm(npy_files, desc="Checking .npy files"):
    try:
        # 尝试加载 .npy 文件
        data = np.load(file_path)
        #print(f"Loaded {file_path} successfully with shape {data.shape}")
    except Exception as e:
        # 捕获并打印加载文件时的错误
        print(f"Error loading {file_path}: {e}")

print("Check completed.")