import pandas as pd
import numpy as np

train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True, sort=False)

numeric_train = train.select_dtypes(include=[np.number])

# 计算相关系数
corr_target = numeric_train.corr()['移动房车险数量']

# 选择相关性绝对值 >= 0.01 的特征
important_feature = corr_target[np.abs(corr_target) >= 0.01].index.tolist()

train = train[important_feature]
test = test[important_feature]

train.to_csv('./Data/train_preprocess.csv', encoding='utf_8_sig', index=False)
test.to_csv('./Data/test_preprocess.csv', encoding='utf_8_sig', index=False)