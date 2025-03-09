import pandas as pd
train = pd.read_excel('./Data/data.xlsx')
test = pd.read_excel('./Data/eval.xlsx')
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
train = data.loc[data['source'] == "train"]
test = data.loc[data['source'] == "test"]
train.drop(['source'], axis=1, inplace=True)
skewness = train.iloc[:, :-1].apply(lambda x: x.skew())
skewness = skewness.sort_values(ascending=False)
print(skewness.head(15))