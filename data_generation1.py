# -*- coding: utf-8 -*-
"""
==========================================
pipeline1-data_generation
第一步，生成一维随机数据，并对其做小波变换，存储成文件
==========================================
"""

import numpy as np
from wavelet.wavelet import wavelet
from utils.utils import f1calc

def generate_X(data_size, abnormal_rate, miu, sigma, miu_ab, sigma_ab):
    abnormal_size = int(data_size * abnormal_rate)
    normal_size = data_size - abnormal_size

    normal_set = np.random.normal(miu, sigma, (normal_size,))
    abnormal_set = np.random.normal(miu_ab, sigma_ab, (abnormal_size,))
    return np.concatenate((normal_set, abnormal_set),axis=0), normal_set, abnormal_set

def generate_Y(data_size, abnormal_rate):
    abnormal_size = int(data_size / 4 * abnormal_rate)
    normal_size = int(data_size / 4 - abnormal_size)

    normal_set = np.zeros(normal_size)
    abnormal_set = np.ones(abnormal_size)
    return np.concatenate((normal_set, abnormal_set),axis=0)

def generateData(TRAIN, TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL, LEVEL):
    ####### 训练集 #######  Generate train data
    X_train, X_train_normal, X_train_abnormal = generate_X(TRAIN, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
    Y_train = generate_Y(TRAIN, ABNORMAL_RATE)

    ####### 测试集 ####### Generate some novel observations，这里注意和训练集是独立同分布的
    X_test, X_test_normal, X_test_abnormal = generate_X(TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
    Y_test = generate_Y(TEST, ABNORMAL_RATE)

    print("正在对训练数据做小波变换")
    X_train_output, y_train_baseline = wavelet(X_train, LEVEL)
    print("正在对测试数据做小波变换")
    X_test_output, y_test_baseline = wavelet(X_test, LEVEL)

    print("小波变换后训练集总样本点：", X_train_output.shape[0], "，训练集异常样本点：", Y_train[Y_train == 1].size, "，占比：", round(Y_train[Y_train == 1].size / X_train_output.shape[0], 2))
    print("小波变换后测试集总样本点：", X_test_output.shape[0], "，训练集负样本点：", Y_test[Y_test == 1].size, "，占比：", round(Y_test[Y_test == 1].size / X_test_output.shape[0], 2))


    # 统计精度，召回率
    recall, precision, F1, acc = f1calc(Y_train, y_train_baseline)
    print ("baseline train set recall: ", recall,"; precision:", precision, "F1-score:", F1, "acc:", acc)
    recallTe, precisionTe, F1Te, accTe = f1calc(Y_test, y_test_baseline)
    print ("baseline test set recall: ", recallTe,"; precision:", precisionTe, "F1-score:", F1Te, "acc:", accTe)

    print("保存文件: ../data/simulate/X_train.npy")
    np.save('../data/simulate/X_train.npy', X_train_output )
    print("保存文件: ../data/simulate/Y_train.npy")
    np.save('../data/simulate/Y_train.npy', Y_train)
    print("保存文件: ../data/simulate/X_test.npy")
    np.save('../data/simulate/X_test.npy', X_test_output )
    print("保存文件: ../data/simulate/Y_test.npy")
    np.save('../data/simulate/Y_test.npy', Y_test)