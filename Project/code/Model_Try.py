# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:04:40 2021
@author: Ricky
"""
import numpy as np
import pandas as pd
from numpy import log as ln

train_csv = pd.read_csv('./trainB2.csv')
test_csv = pd.read_csv('./testB2.csv')

train_data = np.array(train_csv.drop(columns=['PassengerId']))
test_data = np.array(test_csv.drop(columns=['PassengerId']))

print(pd.DataFrame(train_data).isnull().sum())
print(pd.DataFrame(test_data).isnull().sum())

X_train = train_data[:,1:]
y_train = train_data[:,0]
X_test = test_data

#%% KNN
K_neighbors = 5
X_train = ( X_train - np.mean(X_train, axis=0) ) / np.std(X_train, axis=0, ddof=1)
X_test = ( X_test - np.mean(X_test, axis=0) ) / np.std(X_test, axis=0, ddof=1)

def KNN(X_train, y_train, X_test, k):
    y_pred = np.zeros(len(X_test))
    distances = pd.DataFrame(np.zeros([len(X_train),2]), columns=['y','dist'])
    distances['y'] = y_train
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            distances['dist'][j] = np.sqrt(np.sum((X_test[i]-X_train[j])**2))
        dist = distances.sort_values(by=['dist'])[:k]
        y_pred[i] = dist['y'].value_counts().idxmax()
    return y_pred

y_pred1 = KNN(X_train, y_train, X_test, k=K_neighbors)

#%% Gaussian NB
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

y_pred2 = Gaussian_NB(X_train, y_train, X_test)

#%% export csv
y_pred1 = y_pred1.astype(int)
y_pred2 = y_pred2.astype(int)
PassengerId = pd.DataFrame(np.arange(892,1310,1), columns=['PassengerId'])
Survived1 = pd.DataFrame(y_pred1, columns=['Survived'])
Survived2 = pd.DataFrame(y_pred2, columns=['Survived'])
Prediction1 = pd.concat([PassengerId, Survived1], axis=1)
Prediction2 = pd.concat([PassengerId, Survived2], axis=1)
Prediction1.to_csv('C:/Users/Ricky/Desktop/Team_12_K.csv', index=False)
Prediction2.to_csv('C:/Users/Ricky/Desktop/Team_12_G.csv', index=False)
