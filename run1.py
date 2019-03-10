#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:10:32 2019

@author: Diana
"""

print(__doc__)
from data_generation1 import generateData
from svm.svm1 import doSvm
from pca.plot_pca_wavelet import plot_pca

####### 一些参数 #######
TRAIN = 5000
TEST = 5000
# 异常数据占比
ABNORMAL_RATE = 0.1
# 正常数据的均值、方差
MIU = 0
SIGMA = 0.1
# 异常数据的均值、方差
MIU_ABNORMAL = 0.5
SIGMA_ABNORMAL = 0.1
# 画图时的上下界
XMIN = -3.0
YMIN = -3.0
XMAX = 3.0
YMAX = 3.0

# 小波变换的级数
LEVEL = 2

generateData(TRAIN, TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
doSvm(ABNORMAL_RATE, XMIN, YMIN, XMAX, YMAX)
plot_pca()