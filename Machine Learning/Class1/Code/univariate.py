
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS']  
rcParams['axes.unicode_minus'] = False  
# 1. 加载数据
data = pd.read_csv('data/regress_data1.csv')

# 假设数据集中列名为 "人口" 和 "收益"
X = data["人口"].values
y = data["收益"].values

# 数据可视化
plt.scatter(X, y, color='blue', label='原始数据')
plt.xlabel("人口")
plt.ylabel("收益")
plt.title("原始数据分布")
plt.legend()
plt.show()

# 2. 定义损失函数
def compute_loss(w, X, y):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项
    y_pred = X_b.dot(w)
    loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return loss

# 3. 批量梯度下降法
def batch_gradient_descent(X, y, learning_rate, epochs):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项
    w = np.zeros(X_b.shape[1])  # 参数初始化为0
    loss_history = []

    for epoch in range(epochs):
        y_pred = X_b.dot(w)
        gradient = (1 / m) * X_b.T.dot(y_pred - y)
        w -= learning_rate * gradient
        loss = compute_loss(w, X, y)
        loss_history.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, loss_history

# 4. 使用梯度下降法优化
learning_rate = 0.01
epochs = 1000

w_opt, loss_history = batch_gradient_descent(X, y, learning_rate, epochs)

print(f"优化结束时的损失值: {loss_history[-1]:.4f}")
print(f"模型参数:",w_opt)

# 5. 绘制拟合直线与原始数据
plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X, w_opt[0] + w_opt[1] * X, color='red', label='拟合直线')
plt.xlabel("人口")
plt.ylabel("收益")
plt.title("线性回归拟合效果")
plt.legend()
plt.show()

# 绘制损失值变化
plt.plot(range(epochs), loss_history, label="损失值")
plt.xlabel("迭代轮数")
plt.ylabel("损失值")
plt.title("损失值变化曲线")
plt.legend()
plt.show()