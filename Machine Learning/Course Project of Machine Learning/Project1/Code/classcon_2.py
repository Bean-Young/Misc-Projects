import numpy as np
import pandas as pd
from sklearn import tree
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
    train = pd.read_excel('./Data/data.xlsx')
    test = pd.read_excel('./Data/eval.xlsx')
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True, sort=False)

    train_up = train[train['移动房车险数量'] == 1]
    train_down = train[train['移动房车险数量'] == 0]
    train_up = resample(train_up, n_samples=696, random_state=0)
    train_down = resample(train_down, n_samples=1095, random_state=0)
    train = shuffle(pd.concat([train_up, train_down]))

    data = pd.concat([train, test], ignore_index=True, sort=False)

    numeric_train = train.select_dtypes(include=[np.number])

    # 计算相关系数
    corr_target = numeric_train.corr()['移动房车险数量']

    # 选择相关性绝对值 >= 0.01 的特征
    important_feature = corr_target[np.abs(corr_target) >= 0.01].index.tolist()
    train = train[important_feature]
    test = test[important_feature]
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    # 设定不同参数的模型
    models = {
        "Original ID3": tree.DecisionTreeClassifier(criterion='entropy', max_depth=5),
        "ID3 (max_features=20)": tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=20),
        "ID3 (min_samples_split=50)": tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=50),
        "ID3 (min_samples_leaf=50)": tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=50)
    }

    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train_x, train_y)
        accuracy, precision, recall, f1, auc = evaluate_model(model, test_x, test_y)
        print(f"\n{name} Results:")
        print(f'Accuracy  %.4f' % accuracy)
        print(f'Precision %.4f' % precision)
        print(f'Recall    %.4f' % recall)
        print(f'F1 Score  %.4f' % f1)
        print(f'AUC       %.4f' % auc)