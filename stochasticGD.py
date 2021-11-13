import os

import pandas as pd
import math
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df_bow = pd.read_csv('./processed/bagOfWords.csv')
y_target = df_bow['HAM']

df_bow = df_bow.drop(columns=['HAM'])

x_train, x_test, y_train, y_test = train_test_split(df_bow, y_target, test_size=0.3)

params = {
    "loss" : ["log"],
    # "c": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2"],
}

model = SGDClassifier(max_iter=300)
clf = GridSearchCV(model, param_grid=params)
clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)

print('Bag of words \n')
print(clf.best_estimator_)
print(clf.best_params_)
print(f'Accuracy: {clf.best_score_}')

df_bern = pd.read_csv('./processed/bernoulli.csv')
y_target_bern = df_bern['HAM']

df_bern = df_bern.drop(columns=['HAM'])

x_train_bern, x_test_bern, y_train_bern, y_test_bern = train_test_split(df_bern, y_target_bern, test_size=0.3)

clf.fit(x_train_bern, y_train_bern)
y_predicted_bern = clf.predict(x_test_bern)

print('\nBernoulli \n')
print(clf.best_estimator_)
print(clf.best_params_)
print(f'Accuracy: {clf.best_score_}')
