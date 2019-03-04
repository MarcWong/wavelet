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

def split_X(data, abnormal_rate):
    abnormal_size = int(data.shape[0] * abnormal_rate)
    normal_size = data.shape[0] - abnormal_size
    return data[:normal_size,:], data[normal_size + 1:,:]

def doSvm(ABNORMAL_RATE, XMIN, YMIN, XMAX, YMAX):
    ####### 文件读取 #######
    X_train = np.load('../data/simulate/X_train.npy')[:, np.array([0,2])] 
    X_test = np.load('../data/simulate/X_test.npy')[:, np.array([0,2])] 
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

    # 统计正常训练集上的表现
    cntRegular = 0
    for i in range(0, TEST):
        # 判断预测值与真值是否一致
        if (Y_test[i] == y_pred_test[i]):
            cntRegular += 1

    ####### 画图 #######
    # 网格的粒度是第三个参数
    xx, yy = np.meshgrid(np.linspace(XMIN, XMAX, 1000), np.linspace(YMIN, YMAX, 1000))

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure("svm")
    plt.title("2-D SVM")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    b1 = plt.scatter(X_train_normal[:, 0], X_train_normal[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_train_abnormal[:, 0], X_train_abnormal[:, 1], s=s)
    b3 = plt.scatter(X_test_normal[:, 0], X_test_normal[:, 1], s=s)
    b4 = plt.scatter(X_test_abnormal[:, 0], X_test_abnormal[:, 1], s=s)

    plt.axis('tight')

    # X, Y的显示上下界在这里修改
    plt.xlim((XMIN, XMAX))
    plt.ylim((YMIN, YMAX))
    plt.legend([a.collections[0], b1, b2, b3, b4],
            ["learned frontier", "normal train", "abnormal train",
                "normal test", "abnormal test",],
            loc="upper left",
            prop=matplotlib.font_manager.FontProperties(size=8))
    plt.xlabel(
        "error train: %d/%d ; errors test: %d/%d ; "
        % (n_error_train, TRAIN, n_error_test, TEST))

    plt.text(
        -0.1, -0.8,
        "test set correctness: %.2f"
        % (cntRegular / TEST))
    plt.show()
