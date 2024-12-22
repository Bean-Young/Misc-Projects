import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 生成 make_moons 数据集
X, y = make_moons(n_samples=1000, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 方法一：基于 AdaBoostClassifier 实现 Boosting
adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_pred)

# 方法二：基于 GradientBoostingClassifier 实现 Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

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

# 绘制两种 Boosting 方法的决策边界
plot_decision_boundary(adaboost, X_test, y_test, "AdaBoost 分类器的决策边界", "adaboost_decision_boundary.png")
plot_decision_boundary(gb_clf, X_test, y_test, "GradientBoosting 分类器的决策边界", "gradient_boosting_decision_boundary.png")

# 输出分类结果的准确率
print(f"AdaBoost 分类器在 make_moons 测试集上的准确率: {adaboost_accuracy * 100:.2f}%")
print(f"GradientBoosting 分类器在 make_moons 测试集上的准确率: {gb_accuracy * 100:.2f}%")