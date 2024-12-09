from sklearn.datasets import make_moons
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import rcParams
import os 
rcParams['font.sans-serif'] = ['Arial Unicode MS']  
rcParams['axes.unicode_minus'] = False  
output_dir = "/Users/youngbean/Documents/Github/Misc-Projects/Machine Learning/Class3/Output"

# 1. 生成非线性数据
X, y = make_moons(n_samples=100, noise=0.2, random_state=0)

# 2. 绘制生成的非线性数据的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='类别 0', edgecolor="k")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='类别 1', edgecolor="k")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title("非线性数据分布")
plt.legend()
plt.savefig(os.path.join(output_dir, "非线性数据分布.png"))
plt.close()

# 3. 不同核函数的SVM分类实验
kernels = ["linear", "poly", "rbf"]
parameters = {
    "linear": [{"C": 1}],
    "poly": [{"C": 1, "degree": 2}, {"C": 1, "degree": 3}, {"C": 1, "degree": 5}],
    "rbf": [{"C": 1, "gamma": 0.5}, {"C": 1, "gamma": 1.0}, {"C": 1, "gamma": 1.5}]
}

# 遍历每种核函数及其对应的参数
for kernel in kernels:
    for params in parameters[kernel]:
        # 训练SVM模型
        svm = SVC(kernel=kernel, **params)
        svm.fit(X, y)

        # 绘制决策平面
        plt.figure(figsize=(8, 6))
        DecisionBoundaryDisplay.from_estimator(
            svm, X, alpha=0.5, cmap=plt.cm.coolwarm, grid_resolution=500,response_method="predict",
        )
        
        # 绘制原始数据点
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='类别 0', edgecolor="k", marker='o')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='类别 1', edgecolor="k", marker='o')
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        param_text = ", ".join([f"{k}={v}" for k, v in params.items()])
        plt.title(f"SVM 决策平面 (核函数: {kernel}, 参数: {param_text})")
        plt.legend()

        # 保存图像
        filename = f"SVM_决策平面_{kernel}_{param_text.replace('=', '').replace(', ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()