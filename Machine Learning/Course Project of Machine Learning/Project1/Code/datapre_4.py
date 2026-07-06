import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 选择合适的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True, sort=False)

# 选择数值列
num_columns = train.select_dtypes(include=['number']).columns.tolist()
num_columns.remove('移动房车险数量') 
save_dir = './boxplots'

# 逐个绘制并保存
for col in num_columns:
    plt.figure(figsize=(6, 4))
    train.boxplot(column=col, by='移动房车险数量')
    plt.xlabel('移动房车险数量')
    plt.ylabel(col)
    plt.title(col)

    # 保存图片
    save_path = os.path.join(save_dir, f'{col}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  
