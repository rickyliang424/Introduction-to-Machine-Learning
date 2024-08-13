# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:41:03 2021
@author: Ricky
"""
import numpy as np
import pandas as pd

train_csv = pd.read_csv('./train.csv')

train_csv['Fare'] = train_csv['Fare'].fillna(np.mean(train_csv['Fare']))
train_csv['relatives'] = train_csv['SibSp'] + train_csv['Parch']
train_csv['fare_pp'] = train_csv['Fare'] / (train_csv['relatives'] + 1)
print(train_csv.isnull().sum())

train_csv['Pclass'] = train_csv['Pclass'].fillna(0.)
index1 = np.array(train_csv[(train_csv['Pclass'] == 0.) & (train_csv['fare_pp'] >= 20.)].index)
index2 = np.array(train_csv[(train_csv['Pclass'] == 0.) & (train_csv['fare_pp'] > 10.) & (train_csv['fare_pp'] < 20.)].index)
index3 = np.array(train_csv[(train_csv['Pclass'] == 0.) & (train_csv['fare_pp'] <= 10.)].index)

pclass = np.array(train_csv['Pclass'])
for i in range(len(index1)):
    pclass[index1[i]] = 1.
for i in range(len(index2)):
    pclass[index2[i]] = 2.
for i in range(len(index3)):
    pclass[index3[i]] = 3.

train_csv['Pclass'] = pd.DataFrame(pclass)
