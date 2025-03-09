import pandas as pd
train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
nan_count = data.isnull().sum().sort_values(ascending=False)
nan_ratio = nan_count/len(data)
nan_data = pd.concat([nan_count, nan_ratio], axis=1, keys=['count', 'ratio'])
print(nan_data.head(10))