# -*- coding: utf-8 -*-

import numpy as np
import pywt
from EWMA0 import EWMA
import matplotlib.pyplot as plt

filename = '../data/2778_20170127-07-zs'
raw_data_name = str(filename) + '.csv'
raw_data = np.loadtxt(raw_data_name, delimiter=",", skiprows=1)

p = 1
j = 2

for x in range(6,42):
    # 取某一个车厢的传感器
    data = raw_data[:,x]

    mu = np.mean(data)
    sigma = np.var(data)
    wavelet_coefficient = pywt.wavedec(data,'haar',level=j)

    mu_A2 = np.mean(wavelet_coefficient[0])
    sigma_A2 = np.var(wavelet_coefficient[0])
    mu_D2 = np.mean(wavelet_coefficient[1])
    sigma_D2 = np.var(wavelet_coefficient[1])
    mu_D1 = np.mean(wavelet_coefficient[2])
    sigma_D1 = np.var(wavelet_coefficient[2])


    # plt.figure(1)
    # plt.rcParams['figure.dpi'] = 100
    # plt.plot(range(0,data.size),data,linewidth = 0.4,c='black')
    # plt.ylabel('data')
    # plt.xlabel('time/s')
    # plt.show()

    old_data = list(data[0:4])
    wavelet_coefficient = pywt.wavedec(old_data,'haar',level=j)

    for i in range(0,3):
        wavelet_coefficient[i] = list(wavelet_coefficient[i])


    lamda_A2 = 0.4
    lamda_D2 = 0.4
    lamda_D1 = 0.4
    ctrlcA2 = EWMA(lamda_A2,mu_A2,sigma_A2,2,0.15,-0.15)
    ctrlcA2.update(wavelet_coefficient[0])
    ctrlcD2 = EWMA(lamda_D2,mu_D2,sigma_D2,2,0.15,-0.15)
    ctrlcD2.update(wavelet_coefficient[1])
    ctrlcD1 = EWMA(lamda_D1,mu_D1,sigma_D1,1,0.15,-0.15)
    ctrlcD1.update(wavelet_coefficient[2])
    ctrlcht = [ctrlcA2,ctrlcD2,ctrlcD1]





    test = []
    i = 0
    n = 4
    while n < 1024:  
        new_data = data[n]  #新数据
        i = i + 1
        n = n + 1
        for m in range(1,j+1):  #从最低层开始看，是否整除2^m
            if i % 2**m == 0:  #如果整除，说明这一层可以更新一个系数
                temp = old_data[len(old_data)-2**m+1:len(old_data)]
                temp.append(new_data)
                #temp = np.insert(temp,2**m-1,values=new_data,axis = 0)  #这个系数由新数据以及前面的2^m-1个数据共同产生
                new_coefficient = pywt.wavedec(temp,'haar',level=m) #四个变量分别计算这层小波系数，注意，每个变量只能得到1个新系数
                detail_new_coefficient = new_coefficient[1]  #细节层所需的便是第m层，这一层一定在第2个位置(第一个位置为近似层)
                wavelet_coefficient[j-m+1].append(detail_new_coefficient[0]) #更新系数
                ctrlcht[j-m+1].update(detail_new_coefficient) #更新EWMA
                #ctrlcht[j-m+1].plot() #绘制控制图
                if m == j: #如果达到了最高分解层数，那么近似层也要更新，下面道理与上面类似
                    approximate_new_coefficient = new_coefficient[0]
                    wavelet_coefficient[0].append(approximate_new_coefficient[0])
                    ctrlcht[0].update(approximate_new_coefficient)
                    test.append(approximate_new_coefficient)
                    #ctrlcht[0].plot()
        old_data.append(new_data) #将“新数据”作为“老数据”
        if i == 2**j:  #如果达到了2^j，则归0(只需计算除2^j的余)
            i = 0

    for i in range(0,j+1):
        plt.rcParams['figure.dpi'] = 100
        # ctrlcht[i].plot()

    output_name = str(filename) + '/' + str(x - 5) + '.npy'
    np.save(output_name, wavelet_coefficient[0] + wavelet_coefficient[1] + wavelet_coefficient[2] )