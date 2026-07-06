from sklearn.utils import resample, shuffle
import pandas as pd
train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
print(train['移动房车险数量'].value_counts())
train_up = train[train['移动房车险数量'] == 1]
train_down = train[train['移动房车险数量'] == 0]
train_up = resample(train_up, n_samples=696, random_state=0)
train_down = resample(train_down, n_samples=1095, random_state=0)
train = shuffle(pd.concat([train_up, train_down]))
print(train['移动房车险数量'].value_counts())