import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 生成 make_moons 数据集
X, y = make_moons(n_samples=1000, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 定义分类器
knn = KNeighborsClassifier()
log_reg = LogisticRegression(random_state=0)
gnb = GaussianNB()

# 训练分类器
knn.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
gnb.fit(X_train, y_train)

# 手动实现基于多数投票的集成方法
def majority_vote(predictions):
    """
    基于多数投票规则进行集成预测
    :param predictions: 每个分类器的预测结果（二维数组）
    :return: 最终的集成预测结果
    """
    return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

# 收集所有分类器的预测结果
predictions = np.array([knn.predict(X_test), log_reg.predict(X_test), gnb.predict(X_test)])
ensemble_pred_manual = majority_vote(predictions)

# 使用VotingClassifier实现硬投票和软投票
voting_hard = VotingClassifier(estimators=[('knn', knn), ('log_reg', log_reg), ('gnb', gnb)], voting='hard')
voting_soft = VotingClassifier(estimators=[('knn', knn), ('log_reg', log_reg), ('gnb', gnb)], voting='soft')

voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

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

# 绘制每个分类器的决策边界
plot_decision_boundary(knn, X_test, y_test, "KNN 分类器的决策边界", "knn_decision_boundary.png")
plot_decision_boundary(log_reg, X_test, y_test, "逻辑回归的决策边界", "logistic_regression_decision_boundary.png")
plot_decision_boundary(gnb, X_test, y_test, "高斯朴素贝叶斯的决策边界", "gaussian_nb_decision_boundary.png")

# 手写多数投票
class ManualMajorityVote:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        return majority_vote(predictions)

manual_vote_model = ManualMajorityVote([knn, log_reg, gnb])
plot_decision_boundary(manual_vote_model, X_test, y_test, "手写多数投票的决策边界", "manual_majority_vote_decision_boundary.png")

# 硬投票和软投票的决策边界
plot_decision_boundary(voting_hard, X_test, y_test, "硬投票的决策边界", "hard_voting_decision_boundary.png")
plot_decision_boundary(voting_soft, X_test, y_test, "软投票的决策边界", "soft_voting_decision_boundary.png")
# 输出每个模型的准确率
print(f"KNN 分类器的准确率: {knn.score(X_test, y_test) * 100:.2f}%")
print(f"逻辑回归的准确率: {log_reg.score(X_test, y_test) * 100:.2f}%")
print(f"高斯朴素贝叶斯的准确率: {gnb.score(X_test, y_test) * 100:.2f}%")
print(f"手写多数投票的准确率: {((manual_vote_model(X_test) == y_test).mean()) * 100:.2f}%")
print(f"硬投票的准确率: {voting_hard.score(X_test, y_test) * 100:.2f}%")
print(f"软投票的准确率: {voting_soft.score(X_test, y_test) * 100:.2f}%")
