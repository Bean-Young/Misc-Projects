import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc, make_scorer,
    recall_score, accuracy_score, precision_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import interp
import matplotlib.pyplot as plt

# 创建保存图片的目录
if not os.path.exists('./Figure'):
    os.makedirs('./Figure')

# 设置NLS_LANG环境变量
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# 设置pandas显示最大列数
pd.set_option('display.max_columns', 500)

# 读取数据
df = pd.read_csv('./Data/uwide.csv')

# 对SEX列进行因子化处理
factor = pd.factorize(df['SEX'])
df.SEX = factor[0]

# 处理缺失值
null_columns = df.columns[df.isnull().any()]
df = df.fillna(0)

# 创建标签列Sstate
df['Sstate'] = np.where(df['TOTALSCORE'] > 60, 0, 1)

# 选择特征列
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

# 处理样本不平衡问题
df_majority = df[df.Sstate == 0]
df_minority = df[df.Sstate == 1]
count_times = 8
df_majority_downsampled = df_majority
if len(df_majority) > len(df_minority) * count_times:
    new_major_count = len(df_minority) * count_times
    df_majority_downsampled = resample(df_majority, replace=False,
                                       n_samples=new_major_count, random_state=123)
df = pd.concat([df_majority_downsampled, df_minority])

# 划分特征和标签
X = df.iloc[:, 0:len(df.columns.tolist()) - 1].values
y = df.iloc[:, len(df.columns.tolist()) - 1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# 使用SMOTE进行过采样
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
X_train = X_res
y_train = y_res

# 特征标准化
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
grid_search = GridSearchCV(classifier, param_grid, cv=skf, n_jobs=-1, return_train_score=True, scoring=scores, refit=refit_score)
grid_search.fit(X_train, y_train)

# 预测
y_pred = grid_search.predict(X_test)
classifier = grid_search.best_estimator_
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1] 
for f in range(X.shape[1]): 
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure() 
plt.title("Feature importances")
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
plt.bar(range(X.shape[1]),importances[indices],color="r",yerr=std[indices],align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]]) 
plt.savefig('./Figure/feature_importances.png')
plt.clf()

results_importances = list(zip(df.columns[0:len(df.columns.tolist())-1], classifier.feature_importances_))
results_importances.sort(key=lambda x: x[1])
print(results_importances)