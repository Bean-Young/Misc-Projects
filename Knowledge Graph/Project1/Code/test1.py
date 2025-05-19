# 题目 1：读取 CSV 并计算每个类别的平均价格
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./KG-Class/Project1/test.csv')

# 按类别分组并计算平均价格
avg_prices = df.groupby('category')['price'].mean()

# 打印结果
print("每个类别的平均价格：")
print(avg_prices)