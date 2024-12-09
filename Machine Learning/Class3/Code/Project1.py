from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = "/Users/youngbean/Documents/Github/Misc-Projects/Machine Learning/Class3/Output"

# 加载Iris数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

# 选择前两类样本，并取前两个特征（花萼长度和花萼宽度）
X = X[y < 2, :2]
y = y[y < 2]

# 将每类样本分为训练集（前30个样本）和测试集（后20个样本）
X_train, X_test, y_train, y_test = [], [], [], []
for class_label in np.unique(y):
    class_indices = np.where(y == class_label)[0]
    X_train.append(X[class_indices[:30]])
    y_train.append(y[class_indices[:30]])
    X_test.append(X[class_indices[30:]])
    y_test.append(y[class_indices[30:]])

# 将列表转换为数组
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
X_test = np.vstack(X_test)
y_test = np.hstack(y_test)


cmap_points = ['blue', 'red']

# 定义不同的惩罚系数和核函数
penalty_values = [0.01, 0.1, 1.0, 10, 100]
kernels = ["linear", "poly"]

# 存储结果的列表
results = []

# 定义绘图函数
def plot_decision_boundary_with_display(model, X_train, y_train, X_test, y_test, kernel, C, filename):
    # 绘制决策边界
    plt.figure(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(model, X_train, response_method="predict",cmap=plt.cm.coolwarm, alpha=0.5, grid_resolution=500)

    # 绘制训练数据点
    for class_label, color in zip(np.unique(y_train), cmap_points):
        plt.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1],
                    label=f"训练类别 {class_label}", color=color, marker='o', edgecolor="k")

    # 绘制测试数据点
    for class_label, color in zip(np.unique(y_test), cmap_points):
        plt.scatter(X_test[y_test == class_label, 0], X_test[y_test == class_label, 1],
                    label=f"测试类别 {class_label}", color=color, marker='x')

    # 设置标题和标签
    plt.title(f"SVM 决策边界 (核函数: {kernel}, C: {C})")
    plt.xlabel("花萼长度")
    plt.ylabel("花萼宽度")
    plt.legend()

    # 保存图像
    plt.savefig(filename)
    plt.close()

# 遍历不同的核函数和惩罚系数组合
for kernel in kernels:
    for C in penalty_values:
        # 训练SVM模型
        svm = SVC(C=C, kernel=kernel)
        svm.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((kernel, C, accuracy))
        
        # 绘制决策边界并保存
        filename = os.path.join(output_dir, f"SVM_决策边界_{kernel}_C_{C}.png")
        plot_decision_boundary_with_display(svm, X_train, y_train, X_test, y_test, kernel, C, filename)

# 保存实验结果到 CSV
results_df = pd.DataFrame(results, columns=["核函数", "惩罚系数 C", "测试准确率"])
results_df.to_csv(os.path.join(output_dir, "实验结果.csv"), index=False)
