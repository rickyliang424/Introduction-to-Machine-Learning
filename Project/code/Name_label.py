# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:43:32 2021
@author: Ricky
"""
import pandas as pd
import numpy as np

train3 = pd.read_csv("./train3.csv") # 對照檔
csv = pd.read_csv("./train.csv")

csv_Name = list(csv['Name'])
N = pd.DataFrame(0, index=np.arange(len(csv)), columns=['Name'])
n = pd.DataFrame(0, index=np.arange(len(csv)), columns=['Name'])
for i in range(len(csv_Name)):
    N['Name'][i] = csv_Name[i][0]
    n['Name'][i] = ord(csv_Name[i][0])-64

