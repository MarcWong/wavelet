# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:36:38 2018

@author: ASUS-PRO
"""
import numpy as np
import matplotlib.pyplot as plt

class EWMA():
    def __init__(self, lamda = 0.1, mu = 0, sigma = 1, level = 1, ucl = 8.64):
        self.lamda = lamda #参数
        self.value = [] #记录所有的统计量的值
        self.old_number = mu #Z(i-1)
        self.new_number = np.array([]) #Z(i)
        self.mu = mu  #已知的均值
        self.sigma = np.mat(sigma)   #已知的协方差矩阵
        self.ucl = ucl    #控制限
        self.level = level  #对应第几层小波系数
    
    def update(self, data):
        [n,p] = data.shape
        for i in range(0,n):  #EWMA
            self.new_number = self.lamda*data[i,:] + (1-self.lamda)*self.old_number
            s = self.lamda/(2-self.lamda)*(1-(1-self.lamda)**(2*len(self.value)+2))*self.sigma
            s = np.linalg.inv(s)
            trans = self.new_number-self.mu
            self.value.append(np.float(np.dot(np.dot(trans,s),trans.T)))
            self.old_number = self.new_number
    
    def plot(self):   #绘制控制图
        n = len(self.value)
        t_label = np.arange(2**self.level,n*2**self.level+1,2**self.level)  #对应层数把数据对应到原始数量(100个原始数据只能得到50个一层小波系数)
        plt.plot(t_label,self.value,'k-',linewidth = 0.6)
        plt.plot(t_label,self.ucl*np.ones(n),'k--',linewidth = 0.6)
        plt.xlabel('time/s')
        plt.ylabel('MEWMA')
        plt.show()