import numpy as np
import matplotlib.pyplot as plt
import re
import time

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

# Sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义逻辑回归算法
class LogisticRegression:
    def __init__(self, learning_rate=0.003, iterations=100):
        self.learning_rate = learning_rate  # 学习率
        self.iterations = iterations  # 迭代次数

    def fit(self, X, y):
        # 初始化参数
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0
        # 梯度下降
        for i in range(self.iterations):
            # 计算sigmoid函数的预测值, y_hat = w * x + b
            y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
            # 计算损失函数
            #loss = (-1 / len(X)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            dw = (1 / len(X)) * np.dot(X.T, (y_hat - y))
            db = (1 / len(X)) * np.sum(y_hat - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy

def load_dataset():
    X=[]
    y=[]
    with open('./covtype.libsvm.binary.scale.txt', 'r', encoding='utf-8') as f1:
        for num, line in enumerate(f1.readlines()):
            Y=int(line[0:1:1])-1
            line=line[2::]
            line='{'+line[:-12:]+'}'
            line=re.sub(' ',',',line)
            #print(line) (test)
            dic_data=eval(line)
            #Determine if all ten dimensions have values
            if sum(dic_data.keys())==55:
                X.append(list(dic_data.values()))
                y.append(Y)
    import random
    random.seed()
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X1=[]
    y1=[]
    for i in index:
        X1.append(X[i])
        y1.append (y[i])
    X=X1
    y=y1
    X=np.array(X)
    #X=X[:,8:10]
    #print(X)
    y=np.array(y)
    train_x, train_y = X[0:5000], y[0:5000]
    test_x, test_y = X[len(X)-100:], y[len(y)-100:]
    return train_x, train_y, test_x, test_y

def predict1( X):
    global theory_Vw,theory_Vb
    y_hat = sigmoid(np.dot( theory_Vw,X.T) + theory_Vb)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    #print(y_hat)
    return y_hat
def score1(Y,y):
    accuracy = (Y == y).sum() / len(y)
    print("Test set Accuracy*:",accuracy)





def getx(X,y):
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(penalty="l2")
    reg.fit(X, y)
    global  theory_Vw,theory_Vb
    theory_Vw = reg.coef_
    theory_Vb=float(reg.intercept_)

if __name__ == '__main__':
    start=time.time()
    # 划分训练集、测试集
    X_train,y_train, X_test, y_test = load_dataset()
    # 训练模型
    model = LogisticRegression(learning_rate=0.001, iterations=2000)
    model.fit(X_train, y_train)
    #print(y_test)
    # 结果
    getx(X_train,y_train)
    #y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred1=predict1(X_test)
    score1(y_test_pred1,y_test)
    #score_train = model.score(y_train_pred, y_train)
    score_test = model.score(y_test_pred, y_test)
    #print('训练集Accuracy: ', score_train)
    print('Test set Accuracy: ', score_test)

    """

    # 可视化决策边界
    plt.figure()
    x1_min, x1_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    x2_min, x2_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.savefig('Classify(8.9).png')
    #plt.show()

    plt.figure()
    Z_1 = predict1(np.c_[xx1.ravel(), xx2.ravel()])
    Z_1 = Z_1.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z_1, cmap=plt.cm.Spectral)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.savefig('Classify_theory(8.9).png')
    #plt.show()

    """
    end=time.time()
    print('Running time: %s Seconds'%(end-start))