# -*- coding: utf-8 -*-
""" Preprocessing_example """

#%% 1
from sklearn.datasets import load_boston
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
Boston = load_boston()
print(Boston)

#%% 2
print(Boston.keys())

#%% 3
print(Boston.DESCR)

#%% 4
print('Total numbers of feature:', len(Boston.feature_names))
print(Boston.feature_names)

#%% 5
print('Shapes of data:', Boston.data.shape)
print(Boston.data[0:5])

#%% 6
print('Shapes of target:', Boston.target.shape)
print(Boston.target[0:5])

#%% 7
x = pd.DataFrame(Boston.data, columns=Boston.feature_names)
print(x)

#%% 8
y = pd.DataFrame(Boston.target, columns=['target'])
print(y)

#%% 9
# concat:串接兩個數據 ; axis=0為直向合併，axis=1為橫向合併 ; .head()代表取前五row的數據。
data = pd.concat([x,y], axis=1).head()
print(data)

#%% 10
print(data.isnull().sum())

#%% 11
data_1 = data.drop(2, axis=0)
print(data_1)

#%% 12
data_2 = data.drop('ZN', axis=1)
print(data_2)

#%% 13
data_3 = data.loc[:,['AGE','TAX']]
print(data_3)

#%% 14
data_4 = data.iloc[:,[0,1,3,5]]
print(data_4)

#%% 15
add_data = pd.DataFrame(['no','yes','unknown','no','no'], columns=['Haunted'])
print(add_data)

#%% 16
data_5 = pd.concat([data,add_data], axis=1)
print(data_5)

#%% 17
# LabelEncoder():Encode target labels with value between 0 and n_classes-1.
data_6 = data_5
LE = LabelEncoder()
LE.fit(data_6['Haunted'])
data_6['Haunted'] = LE.transform(data_6['Haunted'])
print(data_6)

#%% 18
# train_test_split():Split arrays or matrices into random train and test subsets.
x_data = x.values
y_data = y.values
rate = 0.25
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=rate)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
