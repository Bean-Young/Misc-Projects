import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import tushare as ts
import os
ts.set_token('0990280022ba81ceb7240d3a1ac78dc19453ef4c5c18a7589f6345b4')
pro = ts.pro_api()

print(ts.__version__)
def get_data(code,start,end):
    df=pro.daily(ts_code=code,start_date=start,end_date=end)
    print(df)
    df.index = pd.to_datetime(df.trade_date)
    #设置把日期作为索引
    #df['ma'] = 0.0  # Backtrader需要用到
    #df['openinterest'] = 0.0  # Backtrader需要用到
    #定义两个新的列ma和openinterest
    df = df[['close', 'high', 'low', 'change', 'vol','amount','open']]
    #重新设置df取值，并返回df600893.
    return df
def acquire_code():   #只下载一只股票数据，且只用CSV保存   未来可以有自己的数据库
    inp_code =input("Please enter the stock code:\n")
    inp_start = input("Please enter the start time:\n")
    inp_end = input("Please enter the end time:\n")
    df = get_data(inp_code,inp_start,inp_end)
    print(df.info())
    #输出统计各列的数据量
    print("—"*30)
    #分割线
    print(df.describe())
    #输出常用统计参数
    df.sort_index(inplace=True)
    df.index.name='date'
    #把股票数据按照时间正序排列
    path = os.path.join(os.path.join(os.getcwd(),
    "E:/Python/Project/Informer for Stock Prediction/data"), 'train' + ".csv")
    #os.path地址拼接，''数据地址''为文件保存路径
    # path = os.path.join(os.path.join(os.getcwd(),"数据地址"),inp_code+"_30M.csv")
    df.to_csv(path)
acquire_code()