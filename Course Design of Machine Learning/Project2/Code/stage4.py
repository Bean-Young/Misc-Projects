import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc, make_scorer,
    recall_score, accuracy_score, precision_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import sklearn
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
classifier = RandomForestClassifier(criterion='entropy', oob_score=True)
refit_score = 'precision_score'
# 定义交叉验证策略
skf = StratifiedKFold(n_splits=3)

# 网格搜索
grid_search = GridSearchCV(classifier, param_grid, cv=skf, n_jobs=-1, return_train_score=True, scoring=scores, refit=refit_score)
grid_search.fit(X_train, y_train)

# 预测
y_pred = grid_search.predict(X_test)
print("RandomForest Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicting Negative', 'Predict Positive'], index=['Actual Negative ', 'Actual Positive']))
print("RandomForest accuracy_score:", accuracy_score(y_test, y_pred))
print("RandomForest recall_score:", recall_score(y_test, y_pred))
print("RandomForest roc_auc_score:", sklearn.metrics.roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]))
print("RandomForest f1_score:", sklearn.metrics.f1_score(y_test, y_pred))
# 定义其他算法及其参数网格
# 支持向量机（SVM）对比实验
clf_svm = LinearSVC().fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print("SVM Confusion Matrix:")
print(pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted']))
print("SVM accuracy_score:", sklearn.metrics.accuracy_score(y_test, y_pred_svm))
print("SVM recall_score:", sklearn.metrics.recall_score(y_test, y_pred_svm))
print("SVM roc_auc_score:", sklearn.metrics.roc_auc_score(y_test, y_pred_svm))
print("SVM f1_score:", sklearn.metrics.f1_score(y_test, y_pred_svm))

# 逻辑回归（Logistic Regression）对比实验
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print("\nLogisticRegression accuracy_score:", sklearn.metrics.accuracy_score(y_test, y_pred_lr))
print("LogisticRegression recall_score:", sklearn.metrics.recall_score(y_test, y_pred_lr))
print("LogisticRegression roc_auc_score:", sklearn.metrics.roc_auc_score(y_test, y_pred_lr))
print("LogisticRegression f1_score:", sklearn.metrics.f1_score(y_test, y_pred_lr))

# AdaBoost对比实验
bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=500,
    learning_rate=0.5,
    algorithm="SAMME"
)
bdt_discrete.fit(X_train, y_train)
y_pred_ada = bdt_discrete.predict(X_test)
print("\nAdaBoost accuracy_score:", sklearn.metrics.accuracy_score(y_test, y_pred_ada))
print("AdaBoost recall_score:", sklearn.metrics.recall_score(y_test, y_pred_ada))
print("AdaBoost roc_auc_score:", sklearn.metrics.roc_auc_score(y_test, bdt_discrete.predict_proba(X_test)[:, 1]))
print("AdaBoost f1_score:", sklearn.metrics.f1_score(y_test, y_pred_ada))

# AdaBoost 画图部分
discrete_test_errors = []
for discrete_train_predict in bdt_discrete.staged_predict(X_test):
    discrete_test_errors.append(1. - recall_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, c='black', label='SAMME')
plt.legend()
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')


plt.subplot(132)
# 修改 ylim，让纵坐标从 0 开始
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors, "b", label='SAMME', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_errors.max() * 1.2))  # 修改此处，下限设为 0
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights, "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

plt.subplots_adjust(wspace=0.25)
plt.savefig('./Figure/ada_boost_plot.png')
plt.clf()