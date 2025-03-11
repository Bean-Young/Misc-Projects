import os
import pandas as pd

# 目标文件夹
folder_path = "./boxplots"
rename_log = []  # 记录原文件名和新文件名

# 获取所有 .png 文件
png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# 按原文件名排序，确保顺序
png_files.sort()

# 批量重命名
for i, filename in enumerate(png_files, start=1):
    new_name = f"Figure{i}.png"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)  # 执行重命名
    rename_log.append([filename, new_name])  # 记录重命名规则
    print(f"重命名: {filename} -> {new_name}")

# 存储重命名规则到 Excel
rename_df = pd.DataFrame(rename_log, columns=["原文件名", "新文件名"])
rename_df.to_excel(os.path.join(folder_path, "rename.xlsx"), index=False)

print("批量重命名完成，并已保存重命名规则到 rename.xlsx！")