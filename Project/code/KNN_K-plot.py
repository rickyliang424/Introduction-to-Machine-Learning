# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:20:01 2021
@author: Ricky
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv = pd.read_csv('./trainG.csv')
data = np.array(csv.drop(columns=['PassengerId']))
print(pd.DataFrame(data).isnull().sum())

X = data[:,1:]
y = data[:,0]

X = ( X - np.mean(X, axis=0) ) / np.std(X, axis=0, ddof=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

def calculate_accuracy(y_test, y_pred):
    count = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    accuracy = count / len(y_test)
    return accuracy

k = [1, 3, 5, 7, 9]
accuracy = []
for i in k:
    y_pred = (KNN(X_train, y_train, X_test, k=i))
    accuracy.append(calculate_accuracy(y_pred, y_test))

plt.figure()
plt.plot(k, accuracy)
plt.title('KNN with different k-value')
plt.xlabel('K neighbors')
plt.ylabel('Accuracy')
