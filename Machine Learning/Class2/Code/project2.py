import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS']  
rcParams['axes.unicode_minus'] = False  
# 加载Iris数据集
iris = load_iris()
X = iris.data[:, :2]  # 选择前两个属性 [sepal length, sepal width]
y = iris.target

# 过滤出前两个类别的数据用于二分类
binary_filter = y < 2  # 只保留类别0和1
X_binary = X[binary_filter]
y_binary = y[binary_filter]

# 固定划分：每类取前40个样本作为训练集，后10个作为测试集
X_class_0 = X_binary[y_binary == 0]  # 类别 0 的数据
X_class_1 = X_binary[y_binary == 1]  # 类别 1 的数据

y_class_0 = y_binary[y_binary == 0]  # 类别 0 的标签
y_class_1 = y_binary[y_binary == 1]  # 类别 1 的标签

# 构建训练集
X_train = np.vstack((X_class_0[:40], X_class_1[:40]))
y_train = np.hstack((y_class_0[:40], y_class_1[:40]))

# 构建测试集
X_test = np.vstack((X_class_0[40:50], X_class_1[40:50]))
y_test = np.hstack((y_class_0[40:50], y_class_1[40:50]))

# 修改感知机模型以记录每轮的损失和训练轮数
class PerceptronWithTracking:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.weights = None  # 权重初始化
        self.bias = None  # 偏置初始化
        self.epochs = 0  # 实际训练轮数
        self.loss_history = []  # 损失值记录

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)  # 初始化权重为1
        self.bias = 0  # 初始化偏置为0
        self.epochs = 0
        self.loss_history = []

        for _ in range(self.max_iter):
            errors = 0  # 记录误分类样本数量
            loss = 0  # 本轮的损失
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi  # 更新权重
                self.bias += update  # 更新偏置
                errors += int(update != 0.0)  # 如果有更新，计入误分类
                loss += 0.5 * (target - prediction) ** 2  # 计算二分类损失
            self.loss_history.append(loss / n_samples)  # 平均损失
            self.epochs += 1
            if errors == 0:  # 如果训练集无误分类样本，停止迭代
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)  # 分类阈值：>= 0为1，否则为0

# 定义不同的学习率参数
learning_rates = [0.01, 0.05, 0.1, 0.5]
results = {}

# 运行模型并记录测试正确率和训练轮数
for lr in learning_rates:
    model = PerceptronWithTracking(learning_rate=lr)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    results[lr] = {"accuracy": accuracy, "epochs": model.epochs, "loss_history": model.loss_history}
    print(f"学习率: {lr}, 测试正确率: {accuracy:.2f}, 收敛轮数: {model.epochs}")

# 绘制损失下降图
plt.figure(figsize=(10, 6))
for lr, result in results.items():
    plt.plot(range(1, result["epochs"] + 1), result["loss_history"], label=f"学习率 {lr}")
plt.xlabel("训练轮数 (Epoch)")
plt.ylabel("平均损失值")
plt.title("损失下降图")
plt.legend()
plt.grid()
plt.show()