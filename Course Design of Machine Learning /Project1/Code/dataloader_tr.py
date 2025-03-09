import pandas as pd
data = pd.read_excel('./Data/data.xlsx')
print(data.shape)
print(data.describe())