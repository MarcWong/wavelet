# -*- coding: utf-8 -*-
"""
==========================================
pipeline2-svm
第二步，从文件中读取数据，做svm分解
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from utils.utils import f1calc

def split_X(data, abnormal_rate):
    abnormal_size = int(data.shape[0] * abnormal_rate)
    normal_size = data.shape[0] - abnormal_size
    return data[:normal_size,:], data[normal_size + 1:,:]

def doSvm(ABNORMAL_RATE, XMIN, YMIN, XMAX, YMAX):
    ####### 文件读取 #######
    # X_train = np.load('../data/simulate/X_train.npy')[:, np.array([0,2])] 
    # X_test = np.load('../data/simulate/X_test.npy')[:, np.array([0,2])] 
    X_train = np.load('../data/simulate/X_train.npy')
    X_test = np.load('../data/simulate/X_test.npy')
    Y_train = np.load('../data/simulate/Y_train.npy')
    Y_test = np.load('../data/simulate/Y_test.npy')

    print(X_train.shape)
    print(Y_train.shape)

    # 绘图的点大小
    s = 1
    TRAIN = Y_train.size
    TEST = Y_test.size

    ####### 训练集 #######
    X_train_normal, X_train_abnormal = split_X(X_train, ABNORMAL_RATE)

    print("训练集总样本点：", TRAIN, "，训练集异常样本点：", Y_train[Y_train == 1].size, "，占比：", round(Y_train[Y_train == 1].size / TRAIN, 2))

    ####### 测试集 #######
    X_test_normal, X_test_abnormal = split_X(X_test, ABNORMAL_RATE)

    print("训练集总样本点：", TEST, "，训练集负样本点：", Y_test[Y_test == 1].size, "，占比：", round(Y_test[Y_test == 1].size / TEST, 2))

    ####### svm #######
    # fit the model
    clf = svm.SVC(gamma="auto")
    clf.fit(X_train, Y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    n_error_train = y_pred_train[y_pred_train == 1].size
    n_error_test = y_pred_test[y_pred_test == 1].size

    # 统计精度，召回率
    recall, precision, F1 = f1calc(Y_test, y_pred_test)
    print ("test set recall: ", recall,"; precision:", precision, "F1-score:", F1)
