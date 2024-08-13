# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:18:25 2021
@author: Ricky
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 1
# Dataset Loading & splitting
def random_split_train_test(data, train_frac, test_frac):
    index = np.arange(0, len(data), 1)
    random.shuffle(index)
    
    train_number = round(len(data) * train_frac)
    test_number = round(len(data) * test_frac)
    index_train = index[:train_number]
    index_test = index[-test_number:]
    data_train = pd.DataFrame(np.zeros((train_number, data.shape[1])))
    data_test = pd.DataFrame(np.zeros((test_number, data.shape[1])))
    
    for i in range(data_train.shape[0]):
        for column in range(data_train.shape[1]):
            data_train[column][i] = data[column][index_train[i]]
    for i in range(data_test.shape[0]):
        for column in range(data_test.shape[1]):
            data_test[column][i] = data[column][index_test[i]]
    
    data_train = data_train.sort_values(by=0, ignore_index=True)
    data_test = data_test.sort_values(by=0, ignore_index=True)
    return data_train, data_test

csv = pd.read_csv("./Real_estate.csv")
data = pd.read_csv("./Real_estate.csv", header=None, skiprows=1)
data_train, data_test = random_split_train_test(data, 0.7, 0.3)

#%% 2
# Plot scatter figure including training data and testing data
X2_train = data_train[2]
X2_test = data_test[2]
X3_train = data_train[3]
X3_test = data_test[3]
X4_train = data_train[4]
X4_test = data_test[4]
Y_train = data_train[7]
Y_test = data_test[7]

plt.figure()
plt.plot(X2_train, Y_train, 'o', ms=4, color='blue', label='training data')
plt.plot(X2_test, Y_test, 'o', ms=4, color='red', label='testing data')
plt.xlabel('X2 house age')
plt.ylabel("Y house price of unit area")
plt.title('X2 corresponding to Y')
plt.legend()
plt.grid()

plt.figure()
plt.plot(X3_train, Y_train, 'o', ms=4, color='blue', label='training data')
plt.plot(X3_test, Y_test, 'o', ms=4, color='red', label='testing data')
plt.xlabel('X3 distance to the nearest MRT station')
plt.ylabel("Y house price of unit area")
plt.title('X3 corresponding to Y')
plt.legend()
plt.grid()

plt.figure()
plt.plot(X4_train, Y_train, 'o', ms=4, color='blue', label='training data')
plt.plot(X4_test, Y_test, 'o', ms=4, color='red', label='testing data')
plt.xlabel('X4 number of convenience stores')
plt.ylabel("Y house price of unit area")
plt.title('X4 corresponding to Y')
plt.legend()
plt.grid()

#%% 3
# Define loss function (Mean Square Error)
def LossFunction(Y_train, Y_test, train_pred, test_pred):
    train_loss = 0.0
    test_loss = 0.0
    for i in range(len(Y_train)):
        train_loss = train_loss + (Y_train[i] - train_pred[i])**2
    for i in range(len(Y_test)):
        test_loss = test_loss + (Y_test[i] - test_pred[i])**2
    train_loss = train_loss / len(Y_train)
    test_loss = test_loss / len(Y_test)
    return train_loss, test_loss

#%% 4
# Using gradient method
# Y = 𝛽₀ + 𝛽₁X₂ + 𝛽₂X₃ + 𝛽₃X₄ = (b2 + w2*X₂) + (b3 + w3*X₃) + (b4 + w4*X₄)
X2_data = data[2]
X3_data = data[3]
X4_data = data[4]
Y_data = data[7]

def gradient_descent(X2_data, X3_data, X4_data):
    test_number = 10
    iteration = 7000
    epoch = 100
    B0 = 70  # initial B0
    B1 = 2  # initial B1
    B2 = 2  # initial B2
    B3 = 2  # initial B3
    lr = 1  # learning rate
    lr_B0 = 0
    lr_B1 = 0
    lr_B2 = 0
    lr_B3 = 0
    B0_history = [B0]  # Store initial value of B0 for plotting
    B1_history = [B1]  # Store initial value of B1 for plotting
    B2_history = [B2]  # Store initial value of B2 for plotting
    B3_history = [B3]  # Store initial value of B3 for plotting
    train_loss_history = []
    test_loss_history = []
    for k in range(test_number):
        for i in range(1,iteration+1,1):
            B0_grad = 0.0
            B1_grad = 0.0
            B2_grad = 0.0
            B3_grad = 0.0
            for j in range(len(Y_train)):
                B0_grad = B0_grad-2.0*(Y_train[j]-B0-B1*X2_data[j]-B2*X3_data[j]-B3*X4_data[j])*1.0
                B1_grad = B1_grad-2.0*(Y_train[j]-B0-B1*X2_data[j]-B2*X3_data[j]-B3*X4_data[j])*X2_data[j]
                B2_grad = B2_grad-2.0*(Y_train[j]-B0-B1*X2_data[j]-B2*X3_data[j]-B3*X4_data[j])*X3_data[j]
                B3_grad = B3_grad-2.0*(Y_train[j]-B0-B1*X2_data[j]-B2*X3_data[j]-B3*X4_data[j])*X4_data[j]
            lr_B0 = lr_B0 + B0_grad **2
            lr_B1 = lr_B1 + B1_grad **2
            lr_B2 = lr_B2 + B2_grad **2
            lr_B3 = lr_B3 + B3_grad **2
            B0 = B0 - lr/np.sqrt(lr_B0) * B0_grad  # Updata parameters
            B1 = B1 - lr/np.sqrt(lr_B1) * B1_grad  # Updata parameters
            B2 = B2 - lr/np.sqrt(lr_B2) * B2_grad  # Updata parameters
            B3 = B3 - lr/np.sqrt(lr_B3) * B3_grad  # Updata parameters
            B0_history.append(B0)  # Store parameters for plotting
            B1_history.append(B1)  # Store parameters for plotting
            B2_history.append(B2)  # Store parameters for plotting
            B3_history.append(B3)  # Store parameters for plotting
            if i % epoch == 0:
                train_pred = B0 + B1*X2_train + B2*X3_train + B3*X4_train
                test_pred = B0 + B1*X2_test + B2*X3_test + B3*X4_test
                train_loss, test_loss = LossFunction(Y_train, Y_test, train_pred, test_pred)
                train_loss_history.append(train_loss)
                test_loss_history.append(test_loss)
                print('===== Test %d/10 times.' %(k+1), ' Iteration %d Done =====' %i)
                print('Training loss: %.4f\t' %train_loss,'Testing loss: %.4f.' %test_loss)
    return B0_history, B1_history, B2_history, B3_history, train_loss_history, test_loss_history

def plot_gradient(Y_data, X_data, b_history, w_history):
    x = np.arange(0, 100, 1)  # bias
    y = np.arange(-5, 5, 0.1)  # weight
    Z = np.zeros((len(x), len(y)))
    X, Y = np.meshgrid(x, y)
    for i in range(len(x)):
        for j in range(len(y)):
            b = x[i]
            w = y[j]
            for k in range(len(X_data)):
                Z[j][i] = Z[j][i] + (Y_data[k] - b - w * X_data[k])**2  # L(b,w)=∑[y-(b+w*x)]^2
            Z[j][i] = Z[j][i] / len(X_data)
    plt.figure()
    plt.contourf(X, Y, Z, 100, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.plot(b_history[-1], w_history[-1], 'x', ms=16, markeredgewidth=4, color='orange')
    plt.plot(b_history, w_history, 'o-', ms=3, lw=1, color='black')
    plt.title('Gradient descent', fontsize=16)
    plt.grid()
    return

def plot_model(X_data, X_train, X_test, b, w):
    plt.figure()
    x_range = np.arange(min(X_data), max(X_data), 0.1)
    y_range = b + w * x_range
    plt.plot(X_train, Y_train, 'o', ms=4, color='blue', label='training data')
    plt.plot(X_test, Y_test, 'o', ms=4, color='red', label='testing data')
    plt.plot(x_range, y_range, '-', lw=3, color='black', label='model function')
    plt.ylabel('Y house price of unit area')
    plt.legend()
    plt.grid()
    return

B0_hist, B1_hist, B2_hist, B3_hist, train_loss, test_loss = gradient_descent(X2_train, X3_train, X4_train)
B0 = B0_hist[-1]
B1 = B1_hist[-1]
B2 = B2_hist[-1]
B3 = B3_hist[-1]

plot_gradient(Y_data, X2_data, B0_hist, B1_hist)
plt.xlabel('B0', fontsize=12)
plt.ylabel('B1', fontsize=12)
plot_gradient(Y_data, X3_data, B0_hist, B2_hist)
plt.xlabel('B0', fontsize=12)
plt.ylabel('B2', fontsize=12)
plot_gradient(Y_data, X4_data, B0_hist, B3_hist)
plt.xlabel('B0', fontsize=12)
plt.ylabel('B2', fontsize=12)

plot_model(X2_data, X2_train, X2_test, B0, B1)
plt.xlabel('X2 house age')
plt.title('X2 corresponding to Y with model function')
plot_model(X3_data, X3_train, X3_test, B0, B2)
plt.xlabel('X3 distance to the nearest MRT station')
plt.title('X3 corresponding to Y with model function')
plot_model(X4_data, X4_train, X4_test, B0, B3)
plt.xlabel('X4 number of convenience stores')
plt.title('X4 corresponding to Y with model function')

#%% 5
# Using least square method
# 𝑦 = 𝑋•𝛽  ⇒  𝛽 = (𝑋^𝑇 • 𝑋)^(−1) • 𝑋^𝑇 • 𝑦
def least_square(X2_order, X3_order, X4_order):
    total_variable = X2_order + X3_order + X4_order + 1
    
    # Find all parameters
    X = np.zeros((len(X2_train), total_variable))
    X[:,0] = np.ones(len(X2_train))
    for i in range(1, X2_order+1, 1):
        X[:,i] = (X2_train**i)
    for j in range(1, X3_order+1, 1):
        X[:,X2_order+j] = (X3_train**j)
    for k in range(1, X4_order+1, 1):
        X[:,X2_order+X3_order+k] = (X4_train**k)
    X_t = X.transpose()
    matrix = np.dot(X_t, X)
    matrix_inverse = np.linalg.inv(matrix)
    para = np.dot(matrix_inverse, np.dot(X_t, Y_train))
    
    # Get training and testing loss
    train_pred = para[0]
    test_pred = para[0]
    for i in range(1, X2_order+1, 1):
        train_pred = train_pred + para[i] * (X2_train **i)
        test_pred = test_pred + para[i] * (X2_test **i)
    for j in range(1, X3_order+1, 1):
        train_pred = train_pred + para[X2_order+j] * (X3_train **j)
        test_pred = test_pred + para[X2_order+j] * (X3_test **j)
    for k in range(1, X4_order+1, 1):
        train_pred = train_pred + para[X2_order+X3_order+k] * (X4_train **k)
        test_pred = test_pred + para[X2_order+X3_order+k] * (X4_test **k)
    train_loss, test_loss = LossFunction(Y_train, Y_test, train_pred, test_pred)
    
    # plot X2 data and model
    x_range = np.arange(min(X2_data), max(X2_data), 0.1)
    y_range = para[0]
    for i in range(1, X2_order+1, 1):
        y_range = y_range + para[i] * (x_range**i)
    plt.figure()
    plt.plot(X2_train, Y_train, 'o', ms=4, color='blue', label='training data')
    plt.plot(X2_test, Y_test, 'o', ms=4, color='red', label='testing data')
    plt.plot(x_range, y_range, '-', lw=3, color='black', label='model function')
    plt.xlabel("X2 house age")
    plt.ylabel("Y house price of unit area")
    plt.title("X2 model (Degree=%d)" %(X2_order))
    plt.legend()
    plt.grid()
    
    # plot X3 data and model
    x_range = np.arange(min(X3_data), max(X3_data), 0.1)
    y_range = para[0]
    for j in range(1, X3_order+1, 1):
        y_range = y_range + para[i+j] * (x_range**j)
    plt.figure()
    plt.plot(X3_train, Y_train, 'o', ms=4, color='blue', label='training data')
    plt.plot(X3_test, Y_test, 'o', ms=4, color='red', label='testing data')
    plt.plot(x_range, y_range, '-', lw=3, color='black', label='model function')
    plt.xlabel("X3 distance to the nearest MRT station")
    plt.ylabel("Y house price of unit area")
    plt.title("X3 model (Degree=%d)" %(X3_order))
    plt.legend()
    plt.grid()
    
    # plot X4 data and model
    x_range = np.arange(min(X4_data), max(X4_data), 0.1)
    y_range = para[0]
    for k in range(1, X4_order+1, 1):
        y_range = y_range + para[i+j+k] * (x_range**k)
    plt.figure()
    plt.plot(X4_train, Y_train, 'o', ms=4, color='blue', label='training data')
    plt.plot(X4_test, Y_test, 'o', ms=4, color='red', label='testing data')
    plt.plot(x_range, y_range, '-', lw=3, color='black', label='model function')
    plt.xlabel("X4 number of convenience stores")
    plt.ylabel("Y house price of unit area")
    plt.title("X4 model (Degree=%d)" %(X4_order))
    plt.legend()
    plt.grid()
    return para, train_loss, test_loss

para1, train_loss1, test_loss1 = least_square(X2_order=1, X3_order=1, X4_order=1)
para2, train_loss2, test_loss2 = least_square(X2_order=2, X3_order=1, X4_order=1)
para3, train_loss3, test_loss3 = least_square(X2_order=1, X3_order=2, X4_order=1)
para4, train_loss4, test_loss4 = least_square(X2_order=1, X3_order=1, X4_order=2)
para5, train_loss5, test_loss5 = least_square(X2_order=2, X3_order=2, X4_order=1)
para6, train_loss6, test_loss6 = least_square(X2_order=2, X3_order=1, X4_order=2)
para7, train_loss7, test_loss7 = least_square(X2_order=1, X3_order=2, X4_order=2)
para8, train_loss8, test_loss8 = least_square(X2_order=2, X3_order=2, X4_order=2)
