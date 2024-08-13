# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:45:20 2021

@author: Ricky
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.read_csv('../train3.csv', index_col=None)
print(data1)

# examine the correlation btw features
# seaborn
correlation_matrix=data1.corr().round(2)
fig, ax = plt.subplots(figsize=(8,6))   
sns.heatmap(data=correlation_matrix, annot = True)
