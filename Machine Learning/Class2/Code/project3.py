from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
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

# 使用Scikit-learn中的感知机模型
sklearn_model = Perceptron(tol=0.001, random_state=42)
sklearn_model.fit(X_train, y_train)

# 定义绘制分类结果的函数
def plot_decision_boundary_sklearn(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', edgecolor='black', s=50, label='类别 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', edgecolor='black', s=50, label='类别 1')
    plt.xlabel("花萼长度 (Sepal Length)")
    plt.ylabel("花萼宽度 (Sepal Width)")
    plt.title(title)
    plt.legend()
    plt.show()

# 绘制训练集上的分类结果
plot_decision_boundary_sklearn(sklearn_model, X_train, y_train, "Scikit-learn 训练集分类结果")

# 绘制测试集上的分类结果
plot_decision_boundary_sklearn(sklearn_model, X_test, y_test, "Scikit-learn 测试集分类结果")

# 比较两种模型的结果
# 自己实现的模型在训练集上的分类结果
custom_model = PerceptronWithTracking(learning_rate=0.1)
custom_model.fit(X_train, y_train)
plot_decision_boundary(custom_model, X_train, y_train, "自己实现的模型训练集分类结果")
plot_decision_boundary_sklearn(custom_model, X_test, y_test, "自己实现的模型测试集分类结果")

# 分析产生不同的原因
# 计算测试集上的分类正确率
y_pred_custom = custom_model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)
accuracy_custom = np.mean(y_pred_custom == y_test)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test)

# 打印测试正确率
print(f"自己实现的模型测试正确率: {accuracy_custom:.2f}")
print(f"Scikit-learn模型测试正确率: {accuracy_sklearn:.2f}")

# 打印迭代轮数
print(f"自己实现的模型的迭代轮数: {custom_model.epochs}")
print(f"Scikit-learn模型的迭代轮数: {sklearn_model.n_iter_}")