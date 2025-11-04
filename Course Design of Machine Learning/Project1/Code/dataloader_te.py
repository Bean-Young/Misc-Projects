import pandas as pd
data = pd.read_excel('./Data/eval.xlsx')
print(data.shape)
print(data.describe())