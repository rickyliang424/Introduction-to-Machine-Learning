# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:50:35 2021
@author: Ricky
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data_train.npy")

#%% Method_1 : plt.imsave()
path1 = "C:/Users/Ricky/Desktop/Animals-1/"
for i in range(len(data)):
    fig = data[i,:,:,:]
    filename = str(i) + ".png"
    plt.imsave(path1 + filename, fig)

#%% Method_2 : plt.savefig()
path2 = "C:/Users/Ricky/Desktop/Animals-2/"
for i in range(len(data)):
    fig = data[i,:,:,:]
    plt.imshow(fig)
    plt.axis('off')
    filename = str(i) + ".png"
    plt.savefig(path2 + filename, bbox_inches='tight')
    plt.close()
