import pandas as pd
train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
count = data.apply(lambda x:len(x.unique())).sort_values(ascending=False)
print(count.head(10))