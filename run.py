# -*- coding: utf-8 -*-
"""
==========================================
用此脚本将数据生成、小波变换、svm串在一起
==========================================
"""
print(__doc__)
from data_generation import generateData
from svm.svm import doSvm
from pca.plot_pca_wavelet import plot_pca
import time

####### 一些参数 #######
TRAIN = 5000
TEST = 5000
# 异常数据占比
ABNORMAL_RATE = 0.4
# 正常数据的均值、方差
MIU = 0
SIGMA = 1
# 异常数据的均值、方差
MIU_ABNORMAL = 0.5
SIGMA_ABNORMAL = 1
# 画图时的上下界
XMIN = -1.0
YMIN = -1.0
XMAX = 1.0
YMAX = 1.0

# 小波变换的级数
LEVEL = 7

generateData(TRAIN, TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL, LEVEL)
time_start = time.time()
doSvm(ABNORMAL_RATE, XMIN, YMIN, XMAX, YMAX)
time_end = time.time()
print("svm用时:",time_end - time_start,"秒")

plot_pca()