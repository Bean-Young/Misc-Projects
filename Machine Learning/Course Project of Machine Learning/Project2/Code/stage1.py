from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
import joblib 
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import resample 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from scipy.stats import binned_statistic 
import itertools
import os
os.environ['NLS_LANG']='SMPLFIED CHINESE_CHNA.UTF8'
pd.set_option('display.max_columns',500)
df = pd.read_csv('./Data/uwide.csv')
df.info()
df.describe()
factor = pd.factorize(df['SEX'])
df.SEX = factor[0]
null_columns=df.columns[df.isnull().any()]
print(df[df.isnull().any(axis=1)][null_columns].head())
df = df.fillna(0)
print(df.head())
df['Sstate']=np.where(df['TOTALSCORE']>60,0,1)
print(df.head())
cols = df.columns.tolist()
df = df[[
'BROWSER_COUNT',
'COURSE_COUNT',
'COURSE_SUM_VIEW',
'COURSE_AVG_SCORE',
'EXAM_AH_SCORE',
'EXAM_WRITEN_SCORE',
'EXAM_MIDDLE_SCORE',
'EXAM_LAB',
'EXAM_PROGRESS',
'EXAM_GROUP_SCORE',
'EXAM_FACE_SCORE',
'EXAM_ONLINE_SCORE',
# 'NODEBB_LAST_POST',
'NODEBB_CHANNEL_COUNT',
'NODEBB_TOPIC_COUNT',
'COURSE_SUM_VIDEO_LEN',
'SEX',
# 'MAJORID',
# 'STATUS',
# 'GRADE',
# 'CLASSID',
'EXAM_HOMEWORK',
'EXAM_LABSCORE',
'EXAM_OTHERSCORE',
'NODEBB_PARTICIPATIONRATE',
'COURSE_WORKTIME',
'COURSE_WORKACCURACYRATE',
'COURSE_WORKCOMPLETERATE',
'NODEBB_POSTSCOUNT',
'NODEBB_NORMALBBSPOSTSCOUONT',
'NODEBB_REALBBSARCHIVECOUNT',
'NORMALBBSARCHIVECOUNT',
'COURSE_WORKCOUNT',
# 'STUNO',
# 'ID',
# 'STUID',
# 'COURSEID',
# 'HOMEWORKSCORE',
# 'WRITTENASSIGNMENTSCORE',
# 'MIDDLEASSIGNMENTSCORE',
# 'LABSCORE',
# 'OTHRFRCTCRF',
'Sstate',
# 'STUDYTTI_M',
# 'COURSEID',
# 'LXPAPERID',
# 'PROCESS',
# 'GROUPSTUDYCOUR',
# 'FACESTUDYSCORE',
# 'ONLINESTUDYSCORE'
]]
print(df.columns.tolist())
print(len(df.columns.tolist()))
print(df.Sstate.value_counts())

df_majority = df[df.Sstate == 0]
df_minority = df[df.Sstate == 1]
count_times = 8
df_majority_downsampled = df_majority
if len(df_majority) > len(df_minority) * count_times:
    new_major_count = len(df_minority) * count_times
    df_majority_downsampled = resample(df_majority, replace=False, 
                                       n_samples=new_major_count, random_state=123)
df = pd.concat([df_majority_downsampled, df_minority])
print(df.Sstate.value_counts())

X = df.iloc[:,0:len(df.columns.tolist())-1].values
y = df.iloc[:,len(df.columns.tolist())-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)
print("train count:",len(X_train),"test count:",len(X_test))

from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
X_res,y_res = sm.fit_resample(X_train,y_train)
print("Original dataset shape %s"% Counter(y_train))
print("Resampled dataset shape %s"%Counter(y_res))
X_train = X_res
y_train = y_res

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 定义随机森林分类器和参数网格
param_grid = {
    'min_samples_split': range(2, 10),
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [5, 10, 15, 20],
    'max_features': [5, 10, 20]
}
scores = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}
classifier = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=42)
refit_score = 'precision_score'
# 定义交叉验证策略
skf = StratifiedKFold(n_splits=3)

# 网格搜索
grid_search = GridSearchCV(classifier, param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 预测
y_pred = grid_search.predict(X_test)

# 输出结果
print("最佳参数:", grid_search.best_params_)
print("混淆矩阵:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['预测负类', '预测正类'], index=['实际负类', '实际正类']))
print("准确率:", accuracy_score(y_test, y_pred))
print("召回率:", recall_score(y_test, y_pred))
print("AUC 值:", sklearn.metrics.roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]))
print("F1 值:", sklearn.metrics.f1_score(y_test, y_pred))