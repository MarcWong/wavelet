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

def doSvm(X_train_path, X_test_path, Y_train_path, Y_test_path):
    ####### 文件读取 #######
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    Y_train = np.load(Y_train_path)
    Y_test = np.load(Y_test_path)

    print(X_train.shape)
    print(Y_train.shape)

    ####### svm #######
    # fit the model
    clf = svm.SVC(gamma="auto")
    clf.fit(X_train, Y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    n_error_train = y_pred_train[y_pred_train == 1].size
    n_error_test = y_pred_test[y_pred_test == 1].size

    # 统计精度，召回率
    recall, precision, F1, acc = f1calc(Y_test, y_pred_test, True)
    print ("test set recall: ", recall,"; precision:", precision, "F1-score:", F1, "acc", acc)
