import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample, shuffle

# 评估模型
def evaluate_model(model, test_x, test_y):
    pred_y = model.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y, average='binary')  
    recall = recall_score(test_y, pred_y, average='binary')
    f1 = f1_score(test_y, pred_y, average='binary')
    auc = roc_auc_score(test_y, pred_y)
    
    return accuracy, precision, recall, f1, auc

if __name__ == '__main__':
    # 读取数据
    train = pd.read_excel('./Data/data.xlsx')
    test = pd.read_excel('./Data/eval.xlsx')
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True, sort=False)

    # 处理类别不平衡问题
    train_up = train[train['移动房车险数量'] == 1]
    train_down = train[train['移动房车险数量'] == 0]
    train_up = resample(train_up, n_samples=696, random_state=0)
    train_down = resample(train_down, n_samples=1095, random_state=0)
    train = shuffle(pd.concat([train_up, train_down]))

    data = pd.concat([train, test], ignore_index=True, sort=False)

    # 选择数值特征
    numeric_train = train.select_dtypes(include=[np.number])

    # 计算相关系数
    corr_target = numeric_train.corr()['移动房车险数量']

    # 选择相关性绝对值 >= 0.01 的特征
    important_feature = corr_target[np.abs(corr_target) >= 0.01].index.tolist()
    train = train[important_feature]
    test = test[important_feature]
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    # 超参数列表
    n_estimators_list = [50, 100, 200]
    learning_rate_list = [0.01, 0.1, 1.0]

    # 遍历不同的超参数组合
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            print(f"\nTraining AdaBoost with n_estimators={n_estimators}, learning_rate={learning_rate} ...")

            # 定义 AdaBoost 分类器（使用决策树作为基学习器）
            model = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1),  # 使用简单的决策树作为弱分类器
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=0
            )

            # 训练模型
            model.fit(train_x, train_y)

            # 评估模型
            accuracy, precision, recall, f1, auc = evaluate_model(model, test_x, test_y)
            print(f"\nAdaBoost Results (n_estimators={n_estimators}, learning_rate={learning_rate}):")
            print(f'Accuracy  %.4f' % accuracy)
            print(f'Precision %.4f' % precision)
            print(f'Recall    %.4f' % recall)
            print(f'F1 Score  %.4f' % f1)
            print(f'AUC       %.4f' % auc)