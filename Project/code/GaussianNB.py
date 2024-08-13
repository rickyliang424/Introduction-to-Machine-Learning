# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:14:04 2021
@author: Ricky
"""
import numpy as np
import pandas as pd
from numpy import log as ln
from sklearn.model_selection import train_test_split

csv = pd.read_csv('../train3.csv')
data = np.array(csv.drop(columns=['PassengerId']))
print(pd.DataFrame(data).isnull().sum())

X = data[:,1:]
y = data[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%% From scratch
def split_classes(X_train, y_train):
    C1_count = (y_train == 1).sum()
    C2_count = (y_train == 0).sum()
    X_train_C1 = np.zeros([C1_count, X_train.shape[1]])
    X_train_C2 = np.zeros([C2_count, X_train.shape[1]])
    y_train_C1 = np.zeros([C1_count, 1])
    y_train_C2 = np.zeros([C2_count, 1])
    c1 = 0
    c2 = 0
    for i in range(len(y_train)):
        if y_train[i] == 1.:
            X_train_C1[c1] = X_train[i]
            y_train_C1[c1] = y_train[i]
            c1 = c1 + 1
        else:
            X_train_C2[c2] = X_train[i]
            y_train_C2[c2] = y_train[i]
            c2 = c2 + 1
    return X_train_C1, X_train_C2, y_train_C1, y_train_C2

def Gaussian_NB(X_train, y_train, X_test):
    y_pred = np.zeros(len(X_test))
    X_train_C1, X_train_C2, y_train_C1, y_train_C2 = split_classes(X_train, y_train)
    C1_mean = np.mean(X_train_C1, axis=0)
    C2_mean = np.mean(X_train_C2, axis=0)
    C1_std = np.std(X_train_C1, axis=0, ddof=1)
    C2_std = np.std(X_train_C2, axis=0, ddof=1)
    C1_prob = len(y_train_C1) / len(y_train)
    C2_prob = len(y_train_C2) / len(y_train)
    for i in range(len(X_test)):
        g1 = ln(C1_prob) - 0.5 * np.sum(( (X_test[i] - C1_mean) / C1_std )**2)
        g2 = ln(C2_prob) - 0.5 * np.sum(( (X_test[i] - C2_mean) / C2_std )**2)
        if g1 > g2:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

y_pred1 = Gaussian_NB(X_train, y_train, X_test)

count = 0
for i in range(len(y_test)):
    if y_pred1[i] == y_test[i]:
        count = count + 1
accuracy1 = count / len(y_test)

#%% Use SKlearn model
from sklearn.naive_bayes import GaussianNB

y_pred2 = GaussianNB().fit(X_train, y_train).predict(X_test)

count = 0
for i in range(len(y_test)):
    if y_pred2[i] == y_test[i]:
        count = count + 1
accuracy2 = count / len(y_test)
