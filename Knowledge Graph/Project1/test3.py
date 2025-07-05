# 题目 3：读取 JSON 并计算总价格
import json

# 读取 JSON 文件
with open('./KG-Class/Project1/test.json', 'r') as file:
    data = json.load(file)

# 提取名称和价格，计算总价
total_price = 0
print("产品名称与价格：")
for product in data['products']:
    name = product['name']
    price = product['price']
    total_price += price
    print(f"{name}: ${price}")

# 打印总价格
print(f"\n总价格：${total_price}")