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

# 设置绘图参数
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 9),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# 绘制ROC曲线
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

skf = StratifiedKFold(n_splits=5)  # 书上分成了3份
linetypes = ['--', ':', '-.', '-', '', '0']

i = 0
for train, test in skf.split(X_test, y_test):
    probas_ = grid_search.predict_proba(X_test[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1.5, linestyle=linetypes[i], alpha=0.8,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
         label='Chance', alpha=.6)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $ \pm $ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('FPR', fontsize=20)
plt.ylabel('TPR', fontsize=20)
# plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('./Figure/ROC_curve.png')
plt.clf()

# 绘制准确率曲线
results = grid_search.cv_results_
plt.plot(results['mean_train_accuracy_score'])
plt.plot(results['mean_test_accuracy_score'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['mean_train_accuracy_score', 'mean_test_accuracy_score'], loc='lower right')
plt.savefig('./Figure/accuracy_curve.png')
plt.clf()

# 绘制召回率曲线
plt.plot(results['mean_train_recall_score'])
plt.plot(results['mean_test_recall_score'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['mean_train_recall_score', 'mean_test_recall_score'], loc='lower right')
plt.savefig('./Figure/recall_curve.png')
plt.clf()

# 绘制OOB误差曲线
min_estimators = 1
max_estimators = 149
clf = grid_search.best_estimator_
errs = []
for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i)
    clf.fit(X_train, y_train)
    oob_error = 1 - clf.oob_score_
    errs.append(oob_error)
plt.plot(errs, label='RandomForestClassifier')
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('./Figure/OOB_error_curve.png')
plt.clf()

# 绘制min_samples_split的验证曲线
param_range = range(2, 10)
train_scores, test_scores = validation_curve(
    estimator=grid_search.best_estimator_,  # 传入最佳模型
    X=X_train,
    y=y_train,
    param_name="min_samples_split",  # 需要变化的超参数
    param_range=param_range,  # 超参数取值范围
    cv=3,
    scoring="recall",
    n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve with RandomForestClassifier")
plt.xlabel("min_samples_split")
plt.ylabel("Score")
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('./Figure/validation_curve_min_samples_split.png')
plt.clf()

# 绘制max_depth的验证曲线
param_range = range(2, 20)
train_scores, test_scores = validation_curve(
    estimator=grid_search.best_estimator_,  # 传入最佳模型
    X=X_train,
    y=y_train,
    param_name="max_depth",  # 需要变化的超参数
    param_range=param_range,  # 超参数取值范围
    cv=3,
    scoring="recall",
    n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve with RandomForestClassifier")
plt.xlabel("max_depth")
plt.ylabel("Score")
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('./Figure/validation_curve_max_depth.png')
plt.clf()