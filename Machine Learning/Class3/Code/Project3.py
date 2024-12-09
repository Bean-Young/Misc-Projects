import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS']  
rcParams['axes.unicode_minus'] = False  
# 定义 RBF 核函数
def rbf_kernel(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# 单类 SVM 类
class SVMKernel:
    def __init__(self, kernel=rbf_kernel, C=1.0, gamma=0.5, tol=1e-4, max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j], gamma=self.gamma)

        self.alpha = np.zeros(n_samples)
        self.bias = 0

        # 使用 SMO 算法优化
        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                E_i = np.sum(self.alpha * y * K[:, i]) + self.bias - y[i]
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = np.sum(self.alpha * y * K[:, j]) + self.bias - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.bias - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.bias - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < self.alpha[i] < self.C:
                        self.bias = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1 + b2) / 2

            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break

        self.support_vectors = X[self.alpha > 1e-5]
        self.support_labels = y[self.alpha > 1e-5]
        self.alpha = self.alpha[self.alpha > 1e-5]

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = np.sum(
                self.alpha * self.support_labels *
                np.array([self.kernel(x, sv, self.gamma) for sv in self.support_vectors])
            ) + self.bias
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

# 多分类 SVM 类
class MultiClassSVM:
    def __init__(self, kernel=rbf_kernel, C=1.0, gamma=0.5):
        self.models = {}
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for cls in unique_classes:
            y_binary = np.where(y == cls, 1, -1)
            model = SVMKernel(kernel=self.kernel, C=self.C, gamma=self.gamma)
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        predictions = {}
        for cls, model in self.models.items():
            predictions[cls] = model.predict(X)
        predictions = np.vstack(list(predictions.values())).T
        return np.argmax(predictions, axis=1)

# 加载数据集
iris = load_iris()
X = iris.data  # 使用所有特征
y = iris.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练多分类 SVM
multi_svm = MultiClassSVM(kernel=rbf_kernel, C=1.0, gamma=0.5)
multi_svm.fit(X_train, y_train)

# 预测和评估
y_pred = multi_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"准确率: {accuracy:.2f}%")

# 绘制分类决策边界，仅降维用于可视化
def plot_decision_boundary_with_pca(X, y, model, title="分类决策边界"):
    # 使用 PCA 将数据降到 2 维，仅用于绘图
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 对 PCA 降维后的网格点进行预测
    grid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和样本点
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    X_pca_classes = pca.transform(X)
    plt.scatter(X_pca_classes[:, 0], X_pca_classes[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.show()

# 使用训练后的模型绘制分类决策边界
plot_decision_boundary_with_pca(X_train, y_train, multi_svm, title="训练集分类决策边界（PCA 可视化）")
plot_decision_boundary_with_pca(X_test, y_test, multi_svm, title="测试集分类决策边界（PCA 可视化）")