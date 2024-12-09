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
print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
# 绘制训练数据的分布散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='类别 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='类别 1')
plt.xlabel("花萼长度 (Sepal Length)")
plt.ylabel("花萼宽度 (Sepal Width)")
plt.title("训练数据分布")
plt.legend()
plt.show()

# 定义感知机模型及其随机梯度下降算法
class Perceptron:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.weights = None  # 权重初始化
        self.bias = None  # 偏置初始化

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)  # 初始化权重为1
        self.bias = 0  # 初始化偏置为0

        for _ in range(self.max_iter):
            errors = 0  # 记录误分类样本数量
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi  # 更新权重
                self.bias += update  # 更新偏置
                errors += int(update != 0.0)  # 如果有更新，计入误分类
            if errors == 0:  # 如果训练集无误分类样本，停止迭代
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)  # 分类阈值：>= 0为1，否则为0

# 绘制决策边界的函数
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='类别 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='类别 1')
    plt.xlabel("花萼长度 (Sepal Length)")
    plt.ylabel("花萼宽度 (Sepal Width)")
    plt.title(title)
    plt.legend()
    plt.show()

# 训练感知机模型
perceptron = Perceptron(learning_rate=0.1)
perceptron.fit(X_train, y_train)

# 绘制训练数据的决策边界
plot_decision_boundary(perceptron, X_train, y_train, "训练数据与决策边界")
# 验证测试集中的类别 0 和类别 1 样本数量
print(f"类别 0 测试样本: {X_test[y_test == 0]}")
print(f"类别 1 测试样本: {X_test[y_test == 1]}")
# 测试模型并计算测试集错误率
y_pred = perceptron.predict(X_test)
test_error_rate = np.mean(y_pred != y_test)
print(f"测试集错误率: {test_error_rate:.2f}")

# 绘制测试数据的决策边界
plot_decision_boundary(perceptron, X_test, y_test, "测试数据与决策边界")