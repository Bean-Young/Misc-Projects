import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

    # 超参数组合
    n_estimators_list = [50, 100]
    max_features_list = ['sqrt', 'log2', None]
    max_depth_list = [10, 20, None]
    min_samples_split_list = [2, 5, 10]

    # 遍历不同的超参数组合
    for n_estimators in n_estimators_list:
        for max_features in max_features_list:
            for max_depth in max_depth_list:
                for min_samples_split in min_samples_split_list:
                    print(f"\nTraining Random Forest with n_estimators={n_estimators}, max_features={max_features}, max_depth={max_depth}, min_samples_split={min_samples_split} ...")

                    # 定义随机森林分类器
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_features=max_features,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=0,
                        bootstrap=True  # 自助采样
                    )

                    # 训练模型
                    model.fit(train_x, train_y)

                    # 评估模型
                    accuracy, precision, recall, f1, auc = evaluate_model(model, test_x, test_y)
                    print(f"\nRandom Forest Results (n_estimators={n_estimators}, max_features={max_features}, max_depth={max_depth}, min_samples_split={min_samples_split}):")
                    print(f'Accuracy  %.4f' % accuracy)
                    print(f'Precision %.4f' % precision)
                    print(f'Recall    %.4f' % recall)
                    print(f'F1 Score  %.4f' % f1)
                    print(f'AUC       %.4f' % auc)