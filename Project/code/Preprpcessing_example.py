# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 01:36:33 2021

@author: ESS305
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# =============================================================================
# 看data長怎樣
# =============================================================================
data = pd.read_csv("example.csv")
print(data)

#%%
# =============================================================================
# 關聯性1---畫圖
# =============================================================================

g = sns.FacetGrid(data, col='Survived')    #準備畫框
g.map(plt.hist, 'Age', bins = 20 )         # plt.hist：指定為條形圖 , bins:設定分多少個區間

#%%
# =============================================================================
# 關聯性2---直接寫程式
# =============================================================================
'''
假設
sex       f m m m f f f m f m m
survived  0 1 0 1 0 0 1 1 1 0 1

data[['Sex', 'Survived']].groupby('sex')
---> sex           f            m
     survived  0 0 0 1 1   1 0 1 1 0 1
     
data[['Sex', 'Survived']].groupby('sex').mean()
--->sex              f               m
    survived  (0+0+0+1+1)/5  (1+0+1+1+0+1)/6

sort_values(by='Survived', ascending=False) :依照Survived的值依序排列，ascending=False代表以降序排列
'''

sex_cor = data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(sex_cor)

#%%
# =============================================================================
# 顯示缺失值
# =============================================================================
print(data.isnull().sum())

#%%
# =============================================================================
# 補缺失值(假設用median來補，不一定是正確方式，只是範例)
# =============================================================================

age_median = data['Age'].dropna().median()     #把年齡的缺失值刪掉後再做平均
print('age_median:', age_median)

data['Age'].fillna(age_median, inplace = True) #inplace = True 代表說直接在原資料做填補

print(data.isnull().sum())

#%%
# =============================================================================
# 刪除特徵
# =============================================================================

data = data.drop(['Ticket'], axis = 1) #指定參數 axis = 1 表示要刪除欄位（column）。

#%%
# =============================================================================
# qcut用法+loc用法
# =============================================================================
'''
如果我們今天有一些連續性的數值，可以使用cut&qcut進行離散化
cut函数是利用數值區間將數值分類，qcut則是用分位數。


先創立一個欄位叫做Age_band，用qcut把年齡分成四個區間，每個區間的人數相同。
接著可以用groupby加上mean來看每個區間的存活率
'''

data['Age_band'] = pd.qcut(data['Age'], 4)
Age_band = data[['Age_band', 'Survived']].groupby(['Age_band'], as_index=False).mean().sort_values(by='Age_band', ascending=True)
print(Age_band)

#%%
'''
知道區段分別是多少後，便可以用loc來設區間，可做為模型輸入
'''

#.loc先列後行，中間用逗號（,）分割
#data.loc[          列        ,   行 ]

print('before_age: \n',data['Age'][1:10])

data.loc[ data['Age'] <= 0.829, 'Age'] = 0                          #將 Age這個欄位中的哪一列小於等於0.829的列 設成0
data.loc[(data['Age'] > 0.829) & (data['Age'] <= 20.5), 'Age'] = 1 
data.loc[(data['Age'] > 20.5) & (data['Age'] <= 26.0), 'Age']  = 2
data.loc[(data['Age'] > 26.0) & (data['Age'] <= 31.5), 'Age'] = 3
data.loc[ data['Age'] > 31.5 , 'Age'] = 4 

data['Age'] = data['Age'].astype(int)  #確保設成整數型態
data = data.drop(['Age_band'], axis=1) #現在可以刪掉這個欄位

print('-'*50)
print('after_age: \n',data['Age'][1:10])
#%%
# =============================================================================
# map + dict()的用法
# =============================================================================
'''

對於map來說，它做的事情就是"將原本的值映射(mapping)到另外的值"
因此map不但可以接收一個函數，它也可以接受dictionary或另外一個Series
只要是可以一一對應的就好

因為性別是字串，所以為了將他轉成數字，使用map函式將字串中的詞一一對應到sex裡
'''
#%% 舉例
V = pd.Series([0, 1, 1, 2, 1, 5])
A = {1:11, 2:22, 3:12, 4:50, 5:6}

B = V.map(A)
print(V,B)
#%%
dict_sex = {'female': 1, 'male': 0}

print('before:\n',data.head())
data['Sex'] = data['Sex'].map(dict_sex).astype(int) 
print('-'*50)
print('after:\n',data.head())



