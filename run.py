# -*- coding: utf-8 -*-
"""
==========================================
用此脚本将数据生成、小波变换、svm串在一起
==========================================
"""
print(__doc__)
from data_generation import generateData
from svm import doSvm

####### 一些参数 #######
TRAIN = 10000
TEST = 10000
# 异常数据占比
ABNORMAL_RATE = 0.1
# 正常数据的均值、方差
MIU = 0
SIGMA = 0.1
# 异常数据的均值、方差
MIU_ABNORMAL = 0.1
SIGMA_ABNORMAL = 0.4
# 画图时的上下界
XMIN = -1.0
YMIN = -1.0
XMAX = 1.0
YMAX = 1.0

generateData(TRAIN, TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
doSvm(ABNORMAL_RATE, XMIN, YMIN, XMAX, YMAX)