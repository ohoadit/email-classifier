import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df_bow = pd.read_csv('./processed/bagOfWords.csv')
df_bern = pd.read_csv('./processed/bernoulli.csv')

y_target = df_bow['HAM']

df_bow = df_bow.drop(columns=['HAM'])
df_bern = df_bern.drop(columns=['HAM'])

x_train_bow, x_test_bow, y_train_bow, y_test_bow = train_test_split(df_bow, y_target, test_size=0.3)

lr = LogisticRegression(max_iter=500, C=0.1)
lr.fit(x_train_bow, y_train_bow)

y_hat_bow = lr.predict(x_test_bow)

print('Bag of words \n')
print(f'Precision: {precision_score(y_test_bow, y_hat_bow)*100}')
print(f'Recall: {recall_score(y_test_bow, y_hat_bow)*100}')
print(f'F-1: {f1_score(y_test_bow, y_hat_bow) * 100}')
print(f'Accuracy: {accuracy_score(y_test_bow, y_hat_bow)*100}')

x_train_bern, x_test_bern, y_train_bern, y_test_bern = train_test_split(df_bern, y_target, test_size=0.3)

b_lr = LogisticRegression(max_iter=1000, C=0.1)
b_lr.fit(x_train_bern, y_train_bern)

y_hat_bern = b_lr.predict(x_test_bern)

print('\n Bernoulli \n')
print(f'Precision: {precision_score(y_test_bern, y_hat_bern)*100}')
print(f'Recall: {recall_score(y_test_bern, y_hat_bern)*100}')
print(f'F-1: {f1_score(y_test_bern, y_hat_bern) * 100}')
print(f'Accuracy: {accuracy_score(y_test_bern, y_hat_bern)*100}')