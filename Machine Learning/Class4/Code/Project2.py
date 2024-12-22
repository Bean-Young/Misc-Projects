import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 生成 make_moons 数据集
X, y = make_moons(n_samples=1000, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 方法一：基于 BaggingClassifier 和 DecisionTreeClassifier 实现随机森林
bagging_rf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=0)
bagging_rf.fit(X_train, y_train)
bagging_rf_pred = bagging_rf.predict(X_test)
bagging_rf_accuracy = accuracy_score(y_test, bagging_rf_pred)

# 方法二：使用 RandomForestClassifier 实现随机森林
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
random_forest_pred = random_forest.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

# make_moons 数据集的特征重要性
moon_feature_importances = random_forest.feature_importances_

# 加载 Iris 数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.25, random_state=0)

# 在 Iris 数据集上训练随机森林
random_forest_iris = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest_iris.fit(X_train_iris, y_train_iris)
iris_feature_importances = random_forest_iris.feature_importances_

# 绘制决策边界函数
def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# 绘制两种随机森林方法的决策边界
plot_decision_boundary(bagging_rf, X_test, y_test, "Bagging 随机森林的决策边界", "bagging_rf_decision_boundary.png")
plot_decision_boundary(random_forest, X_test, y_test, "RandomForestClassifier 的决策边界", "random_forest_decision_boundary.png")

# 展示 make_moons 数据集的特征重要性
print("make_moons 数据集的特征重要性:")
print(f"Feature 1: {moon_feature_importances[0]:.4f}, Feature 2: {moon_feature_importances[1]:.4f}\n")

# 展示 Iris 数据集的特征重要性
print("Iris 数据集的特征重要性:")
for i, feature_name in enumerate(iris.feature_names):
    print(f"{feature_name}: {iris_feature_importances[i]:.4f}")

# 输出分类结果的准确率
print(f"Bagging 随机森林在 make_moons 测试集上的准确率: {bagging_rf_accuracy * 100:.2f}%")
print(f"RandomForestClassifier 在 make_moons 测试集上的准确率: {random_forest_accuracy * 100:.2f}%")
