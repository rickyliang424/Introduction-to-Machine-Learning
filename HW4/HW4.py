# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:50:15 2021
@author: Ricky
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv("InterestsSurvey.csv")
data = csv.to_numpy()

## 比對grand_tot_interests是否正確
tot_int_diff = []
for i in range(len(data)):
    tot_int = data.shape[1] - pd.isnull(data[i]).sum() - 2
    if tot_int - csv['grand_tot_interests'][i] != 0:
        tot_int_diff.append(i)

## nan填0，其他填1
data_int = np.array(data[:,2:])
for i in range(data_int.shape[0]):
    for j in range(data_int.shape[1]):
        if str(data_int[i,j]) == str(np.nan):
            data_int[i,j] = 0
        else:
            data_int[i,j] = 1

## 建立正確的tot_interest
data_tot_int = np.array(data[:,1])
for i in range(len(data)):
    data_tot_int[i] = data.shape[1] - pd.isnull(data[i]).sum() - 2

## 找出group種類並label
group_type = []
group_type.append(csv['group'][0])
for i in range(len(csv)):
    if group_type[-1] != csv['group'][i]:
        group_type.append(csv['group'][i])
group_map = {"C": 1, "P": 2, "R": 3, "I": 4}
data_group = np.array(csv['group'].map(group_map))

## 建立新的正確的data
csv1_group = pd.DataFrame(data_group)
csv1_tot_int = pd.DataFrame(data_tot_int / (data.shape[1] - 2))
csv1_int = pd.DataFrame(data_int)
csv1 = pd.concat([csv1_group, csv1_tot_int, csv1_int], axis=1)
csv1.columns = csv.columns
data1 = csv1.to_numpy()

## 去掉太多或太少 NaN 的 interest
int_count = pd.isnull(csv).sum()
int_nan_prob = int_count / len(csv)
int_useful = int_nan_prob[(0.1 < int_nan_prob) & (int_nan_prob < 0.9)]
j = 0
csv2 = pd.concat([csv1_group, csv1_tot_int], axis=1)
csv2.columns = csv.columns[:2]
for i in range(len(csv1)):
    if csv1.columns[i] == int_useful.index[j]:
        csv2 = pd.concat([csv2, csv1[csv1.columns[i]]], axis=1)
        j = j + 1
        if j >= len(int_useful):
            break
data2 = csv2.to_numpy()

## 建立X(input samples)和y(target values)
X = np.array(data2[:,1:], dtype=float)  # data2 <=> data1 ; [:,2:] <=> [:,1:]
y = np.array(data1[:,0], dtype=float)

#%% feature selection
from sklearn.feature_selection import SelectKBest, chi2
subset_slec = SelectKBest(chi2, k=3).fit(X, y)
subset_csv1 = pd.DataFrame(subset_slec.scores_.transpose(), index=csv2.columns[1:], columns=['chi-score'])
subset_csv1 = subset_csv1.sort_values(by=['chi-score'], ascending=False)

subset_X = SelectKBest(chi2, k=3).fit_transform(X, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Subset selection (component = 3)')
ax.scatter(subset_X[:,0], subset_X[:,1], subset_X[:,2], s=100, c=csv1_group)
plt.show()

#%% PCA
from sklearn.decomposition import PCA
plt.figure()
plt.plot(np.cumsum(PCA().fit(X).explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

PCA_3 = PCA(n_components=3)
PCA_3_feature = PCA_3.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('PCA (component = 3)')
ax.scatter(PCA_3_feature[:,0], PCA_3_feature[:,1], PCA_3_feature[:,2], s=2, c=csv1_group)
# ax.view_init(elev=30,azim=30)
plt.show()

PCA_2 = PCA(n_components=2)
PCA_2_feature = PCA_2.fit_transform(X)
plt.figure()
plt.title('PCA (component = 2)')
plt.scatter(PCA_2_feature[:,0], PCA_2_feature[:,1], s=2, c=csv1_group)
plt.show()

#%% LDA (component=3)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis().fit(X, y)
print('LDA explained variance:\n', LDA.explained_variance_ratio_)

LDA_3 = LinearDiscriminantAnalysis(n_components=3)
LDA_3_feature = LDA_3.fit_transform(X, y)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('LDA (component = 3)')
ax.scatter(LDA_3_feature[:,0], LDA_3_feature[:,1], LDA_3_feature[:,2], s=6, c=y)
ax.view_init(elev=50,azim=220)
ax.dist = 8
plt.show()

#%% LDA (component=2)
LDA_2 = LinearDiscriminantAnalysis(n_components=2)
LDA_2_feature = LDA_2.fit_transform(X, y)
plt.figure()
plt.title('LDA (component = 2)')
plt.scatter(LDA_2_feature[:,0], LDA_2_feature[:,1], s=2, c=y)
plt.show()

#%% LDA (component=1)
LDA_1 = LinearDiscriminantAnalysis(n_components=1)
LDA_1_feature = LDA_1.fit_transform(X, y)
plt.figure()
plt.title('LDA (component = 1)')
plt.scatter(LDA_1_feature[:,0], np.zeros(len(data1)), c=data_group)
plt.show()

plt.figure(figsize = (15,9))
plt.title('LDA (component = 1)')
plt.scatter(np.arange(len(data1)), LDA_1_feature[:,0], c=csv1_group)
plt.bar(np.arange(len(data1)), LDA_1_feature[:,0], width=0.2, facecolor='k')
plt.show()

#%% K-means (WCSS)
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

wcss = []
for i in range(1,11):
   model = KMeans(n_clusters=i, init="k-means++").fit(X)
   wcss.append(model.inertia_)
plt.figure(figsize=(5,5))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%% K-means (3D)
for i in range(1,6):
    fig = plt.figure()
    ax = Axes3D(fig, elev=50, azim=220)
    labels = KMeans(n_clusters=i, init='k-means++').fit(LDA_3_feature).labels_
    ax.scatter(LDA_3_feature[:,0], LDA_3_feature[:,1], LDA_3_feature[:,2], c=labels, s=4)
    ax.set_title('Cluster = %d' %(i))
    ax.dist = 8
    plt.show()

#%% K-means (2D)
for i in range(1,6):
    plt.figure()
    labels = KMeans(n_clusters=i, init='k-means++').fit(LDA_2_feature).labels_
    plt.scatter(LDA_2_feature[:,0], LDA_2_feature[:,1], c=labels, s=2)
    plt.title('Cluster = %d' %(i))
    plt.show()

#%% K-means (Ground Truth)
# Plot the 3D ground truth
for i in (2, 3, 4):
    fig = plt.figure()
    ax = Axes3D(fig, elev=50, azim=220)
    labels = KMeans(n_clusters=i, init='k-means++').fit(LDA_3_feature).labels_
    for name, label in [('C',1), ('P',2), ('R',3), ('I',4)]:
        ax.text3D(LDA_3_feature[y == label, 0].mean(), LDA_3_feature[y == label, 1].mean(), 
                  LDA_3_feature[y == label, 2].mean(), name, horizontalalignment='center', 
                  fontsize=12, bbox=dict(alpha=0.5, edgecolor='r', facecolor='w'))
    ax.scatter(LDA_3_feature[:,0], LDA_3_feature[:,1], LDA_3_feature[:,2], c=labels, s=4)
    ax.set_title('Ground Truth in 3D (%d clusters)' %i)
    ax.dist = 8
    plt.show()

# Plot the 2D ground truth
for i in (2, 3, 4):
    plt.figure()
    labels = KMeans(n_clusters=i, init='k-means++').fit(LDA_3_feature).labels_
    for name, label in [('C',1), ('P',2), ('R',3), ('I',4)]:
        plt.text(LDA_2_feature[y == label, 0].mean(), LDA_2_feature[y == label, 1].mean(), name, 
                 horizontalalignment='center', bbox=dict(alpha=0.5, edgecolor='r', facecolor='w'))
    plt.scatter(LDA_2_feature[:,0], LDA_2_feature[:,1], c=labels, s=2)
    plt.title('Ground Truth in 2D (%d clusters)' %i)
    plt.show()

#%% save to .mat
# from scipy.io import savemat
# path = 'C:/Users/Ricky/Desktop/Matlab/'

# dic_LDA = {'LDA_3_feature':LDA_3_feature, 'y':y}
# savemat(path+"LDA.mat", dic_LDA)

# labels = KMeans(n_clusters=4, init='k-means++').fit(LDA_3_feature).labels_
# KM_pred = pd.DataFrame(labels, columns=['KM_pred'])
# KM_ftr = pd.DataFrame(LDA_3_feature, columns=['feature1','feature2','feature3'])
# kmeans = np.array(pd.concat([KM_pred, KM_ftr], axis=1).sort_values(by=['KM_pred']))
# dic_KM = {'KM_pred':kmeans[:,0], 'KM_ftr':kmeans[:,1:]}
# savemat(path+"KMeans.mat", dic_KM)
