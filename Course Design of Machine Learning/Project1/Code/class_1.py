import numpy as np
from sklearn import tree
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_validate
def load_data(path):
    data = pd.read_csv(path, encoding='utf-8')
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y
def build_model(x, y):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x, y)
    return classifier
def test_model(classifier):
    test_x, test_y = load_data('./Data/test_preprocess.csv')
    scores = cross_validate(classifier, test_x, test_y, cv=5, scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
    return scores
if __name__ == '__main__':
    train_x, train_y = load_data('./Data/train_preprocess.csv')
    classifier = build_model(train_x, train_y)
    scores = test_model(classifier)
    print('Accuracy %.4f' % (np.mean(scores['test_accuracy'])))
    print('Precision %.4f' % (np.mean(scores['test_precision'])))
    print('Recall %.4f' % (np.mean(scores['test_recall'])))
    print('F1 %.4f' % (np.mean(scores['test_f1'])))
    print('AUC %.4f' % (np.mean(scores['test_roc_auc'])))