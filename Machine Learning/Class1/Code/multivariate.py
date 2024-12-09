import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS']  
rcParams['axes.unicode_minus'] = False  

# 1. 加载数据
data = pd.read_csv('data/regress_data2.csv')
X = data[['面积', '房间数']].values  # 特征
y = data['价格'].values  # 目标值

# 2. 特征归一化
def feature_normalize(X):
    mu = np.mean(X, axis=0)  # 每列均值
    sigma = np.std(X, axis=0)  # 每列标准差
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_norm, X_mu, X_sigma = feature_normalize(X)
y_norm, y_mu, y_sigma=feature_normalize(y)

# 添加偏置项
m = len(y_norm)
X_b = np.c_[np.ones((m, 1)), X_norm]  # 偏置项

# 3. 定义损失函数
def compute_loss(w, X, y):
    m = len(y)
    y_pred = X.dot(w)
    loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return loss

# 4. 批量梯度下降法
def batch_gradient_descent(X, y, learning_rate, epochs):
    m = len(y)
    w = np.zeros(X.shape[1])  # 初始化参数
    loss_history = []

    for epoch in range(epochs):
        y_pred = X.dot(w)
        gradient = (1 / m) * X.T.dot(y_pred - y)
        w -= learning_rate * gradient
        loss = compute_loss(w, X, y)
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, loss_history

# 5. 训练多变量线性回归模型
learning_rate = 0.01
epochs = 1000

w_opt, loss_history = batch_gradient_descent(X_b, y_norm, learning_rate, epochs)

# 输出最终损失值和模型参数
print(f"优化结束时的损失值: {loss_history[-1]:.4f}")
print(f"模型参数: {w_opt}")

# 6. 绘制训练误差变化曲线
plt.plot(range(epochs), loss_history, label='训练误差')
plt.xlabel("迭代轮数 (epoch)")
plt.ylabel("损失值 (Loss)")
plt.title("训练误差变化曲线")
plt.legend()
plt.grid(True)
plt.show()
