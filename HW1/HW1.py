# -*- coding: utf-8 -*-
""" Homework 1 """

#%% 1
A = []
for i in range(100,501):
    a = i//100
    b = (i-a*100)//10
    c = (i-a*100-b*10)
    if i==(a**3+b**3+c**3):
        A.append(i)
print("100~500中的水仙花數為：",A)

#%% 2
B = [2]
for i in range(3,101):
    j = 1
    while(i%(i-j)!=0):
        j = j+1
        if i-j==1:
            B.append(i)
            break
print("100以內的所有質數：",B)

#%% 3
import numpy as np
i = 0; count = 0
p = np.zeros([504,3])
for a in range(1,10):
    for b in range(1,10):
        for c in range(1,10):
            if a!=b and b!=c and a!=c:
                n = np.array([a,b,c])
                p[i] = n
                i = i + 1
                if a>b and b>c:
                    count = count + 1
print(p)
print("a>b>c的機率為：",count/504)

#%% 4
C = [1, 1]
def Area(x):
    for i in range(2,x):
        C.append(C[i-1]+C[i-2])
    area = C[x-1]*(C[x-1]+C[x-2])
    return area
print("當正方形數量為15時，黃金矩形的面積為：",Area(15))

#%% 5
import numpy as np
import matplotlib.pyplot as plt
def circle(a,b,r):
    x = np.arange(a-r,a+r,0.001)
    y1 = (r**2-(x-a)**2)**0.5+b
    y2 = -(r**2-(x-a)**2)**0.5+b
    plt.figure(figsize=(6,6))
    plt.plot(x,y1,color='blue')
    plt.plot(x,y2,color='blue')
    plt.xlim((-4,12))
    plt.ylim((-4,12))
    ax = plt.gca()  # 獲取當前座標的位置
    ax.spines['right'].set_color('None')  # 去掉座標圖的右邊的spine
    ax.spines['top'].set_color('None')  # 去掉座標圖的上面的spine
    ax.spines['bottom'].set_position(('data',0))  # 設定y=0為x軸
    ax.spines['left'].set_position(('data',0))  # 設定x=0為y軸
    plt.show
    return
circle = circle(5,3,2)

#%% 6
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
path = 'C:/Users/Ricky/Desktop/大三下 課程/機器學習導論/TA1/'
file = path + 'moon.mat'
Data = sio.loadmat(file)
coordinate = Data['moon']
x = np.arange(0,1,0.001)
y = 0.25*x + 0.375
plt.figure(figsize=(8,8))
plt.scatter(coordinate[:,0],coordinate[:,1],s=5)
plt.plot(x,y,color='red',label='y=0.25*x+0.375')
plt.title('107011153')
plt.legend()
plt.show

#%% 7
import numpy as np
import matplotlib.pyplot as plt
def binomial(n,p,k):
    count = 0
    B = np.zeros(n+1)
    N = np.arange(n+1)
    b = np.random.binomial(n,p,k)
    mean = np.mean(b)
    std = np.std(b)
    for i in range(0,k):
        for j in range(0,n+1):
            if b[i]==j:
                B[j]=B[j]+1
    for m in range(round(mean-std),round(mean+std)+1):
        count = count + B[m]
    plt.bar(N,B)
    plt.show
    print("二項分布(","n =",n,",p =",p,")的標準差為：",round(std,4))
    print("落在平均值正負一個標準差內的比例為：",round(count/k*100,4),"%")
    return
plt.figure(1)
binomial(20,0.5,10**6)
plt.figure(2)
binomial(20,0.7,10**6)
plt.figure(3)
binomial(40,0.5,10**6)
