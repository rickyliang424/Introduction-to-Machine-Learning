# -*- coding: utf-8 -*-
""" Homework 2 """

#%% 1
import pandas as pd
import numpy as np

path = "C:/Users/Ricky/Desktop/大三下 課程/機器學習導論/TA2/"
file = path + "hw_data1.csv"
csv = pd.read_csv(file)

#%% 1.(a)
# data_1a = csv.loc[:,['Team','Yellow Cards','Red Cards']]
data_team = pd.DataFrame(csv, columns=['Team'])
data_yellow = pd.DataFrame(csv, columns=['Yellow Cards'])
data_red = pd.DataFrame(csv, columns=['Red Cards'])
data_1a = pd.concat([data_team,data_yellow,data_red], axis=1)

#%% 1.(b)
data_1b = data_1a.sort_values(by=['Red Cards','Yellow Cards'], ascending=False)

#%% 1.(c)
data_1c = np.mean(data_1b['Yellow Cards'], axis=0)

#%% 1.(d)
# data_1d = csv.loc[(csv['Goals']>5),:].sort_values(by='Goals',ascending=False)
data_1d = csv
for i in range(len(data_1d)):
    if data_1d['Goals'][i] <= 5:
        data_1d = data_1d.drop(i, axis=0)
data_1d = data_1d.sort_values(by='Goals', ascending=False)

#%% 1.(e)
data_1e = csv
name = list(csv['Team'])
for i in range(len(csv)):
    if name[i][0] != 'S':
        data_1e = data_1e.drop(i, axis=0)

#%% 1.(f)
data_1f = csv.iloc[:,0:csv.shape[1]-5]

#%% 1.(g)
data_1g = csv.loc[(csv['Team']=='France')|(csv['Team']=='Poland')|(csv['Team']=='Spain'),
                  ['Team','Shooting Accuracy']].sort_values(by='Shooting Accuracy',ascending=False)    

#%% 2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import log as ln
from math import pi as pi

Boston = load_boston()
boston_data = pd.DataFrame(Boston.data, columns=Boston.feature_names)
boston_target = pd.DataFrame(Boston.target, columns=['TARGET'])
boston = pd.concat([boston_data,boston_target], axis=1)

#%% 2.(a)
target_mean = np.mean(boston, axis=0)
for i in range(len(boston)):
    if boston['TARGET'][i] > target_mean['TARGET']:
        boston['TARGET'][i] = 1
    else:
        boston['TARGET'][i] = 0
boston_train, boston_test = train_test_split(boston, test_size=0.3)
boston_train = boston_train.sort_values(by=['TARGET'],ignore_index=True)

count = 0
while(boston_train['TARGET'][count] != 1):
    count = count + 1
train_C1 = boston_train[0:count]
train_C2 = boston_train[count:len(boston_train)]

train_C1_prob = len(train_C1) / len(boston_train)
train_C1_mean = np.mean(train_C1, axis=0).to_numpy()
train_C1_std = np.std(train_C1, axis=0, ddof=1).to_numpy()
train_C2_prob = len(train_C2) / len(boston_train)
train_C2_mean = np.mean(train_C2, axis=0).to_numpy()
train_C2_std = np.std(train_C2, axis=0, ddof=1).to_numpy()

#%% 2.(b) pre.
correct = 0
test_data = boston_test.to_numpy()
data_predict = np.zeros([152,16])
for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]-1):
        a1 = -0.5*ln(2*pi)
        b1 = ln(train_C1_std[j])
        c1 = ((test_data[i,j]-train_C1_mean[j])**2)/(2*(train_C1_std[j])**2)
        d1 = ln(train_C1_prob)
        g1 = a1 - b1 - c1 + d1
        a2 = -0.5*ln(2*pi)
        b2 = ln(train_C2_std[j])
        c2 = ((test_data[i,j]-train_C2_mean[j])**2)/(2*(train_C2_std[j])**2)
        d2 = ln(train_C2_prob)
        g2 = a2 - b2 - c2 + d2
        if g1 > g2:
            data_predict[i,j] = 0
            data_predict[i,13] = data_predict[i,13] + 1
        if g1 < g2:
            data_predict[i,j] = 1
            data_predict[i,14] = data_predict[i,14] + 1
    if data_predict[i,13] > data_predict[i,14]:
        data_predict[i,15] = 0
    if data_predict[i,13] < data_predict[i,14]:
        data_predict[i,15] = 1
    if data_predict[i,15] == test_data[i,13]:
        correct = correct + 1
accuracy_total = correct / len(test_data)

accuracies = np.zeros([1,13])
correct = np.zeros([1,13])
for j in range(data_predict.shape[1]-3):
    for i in range(data_predict.shape[0]):
        if data_predict[i,j] == test_data[i,13]:
            correct[0,j] = correct[0,j] + 1
    accuracies[0,j] = correct[0,j] / len(test_data)
accuracies = pd.DataFrame(accuracies, columns=Boston.feature_names)
# Accuracy(index): 1st:LSTAT(12), 2nd:RM(5), 3rd:AGE(6), 4th:PTRATIO(10).

#%% 2.(b)
# Use LSTAT and RM
correct = 0
for i in range(data_predict.shape[0]):
    if data_predict[i,5] == data_predict[i,12]:
        if data_predict[i,5] == test_data[i,13]:
            correct = correct + 1
    else:
        if data_predict[i,12] == test_data[i,13]:
            correct = correct + 1
accuracy_2 = correct / len(test_data)

#%% 2.(c)
#  Use LSTAT, RM, AGE and PTRATIO
correct = 0
for i in range(data_predict.shape[0]):
    count_0 = 0
    count_1 = 0
    if data_predict[i,5] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
    if data_predict[i,6] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
    if data_predict[i,10] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
    if data_predict[i,12] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
    
    if count_0 > count_1:
        if test_data[i,13] == 0:
            correct = correct + 1
    if count_0 < count_1:
        if test_data[i,13] == 1:
            correct = correct + 1
    if count_0 == count_1:
        if data_predict[i,12] == test_data[i,13]:
            correct = correct + 1
accuracy_4 = correct / len(test_data)

#%% 2.(d)

#%% 2.(e)
