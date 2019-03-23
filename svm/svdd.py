"""
==========================================
SVDD
==========================================
"""
print(__doc__)

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

####### 一些参数 #######
name= ['223-94.npy',
    '223-95.npy',
    '223-96.npy',
    '223-97.npy',
    '223-100-问题.npy',
    '223-101.npy',
    '223-105-问题.npy',
    '223-106.npy',
    '223-107-小问题.npy',
    '223-108.npy',
    '223-109-断.npy',
    '223-111-非规律.npy',
    '223-112-小问题.npy',
    '223-114-非规律.npy',
    '223-115-非规律.npy',
    '223-117.npy',
    '223-118.npy',
    '223-119-问题.npy',
    '223-120.npy',
    '223-121-非规律.npy',
]

for i in range(0, len(name)):

    X_train = np.load('../../silicon_data/' + name[i])

    # raw_data = np.load('../../data/2544_20160907-13-zs/10.npy')[:, np.array([0,2])]


    ####### 训练集 #######
    # X_train = raw_data[: int(raw_data.shape[0] / 2), :]
    # train_size = X_train.shape[0]
    # print(X_train.shape)


    ####### 测试集 #######
    # X_test = raw_data[int(raw_data.shape[0] / 2):, :] 
    # test_size = X_test.shape[0]
    # print(X_test.shape)


    ####### svdd #######
    # fit the model

    clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.001)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    # y_pred_test = clf.predict(X_test)
    n_abnormal_train = y_pred_train[y_pred_train == -1].size
    # n_abnormal_test = y_pred_test[y_pred_test == -1].size

    # print ("abnormal train: %d/%d ; abnormal test: %d/%d ; "
    #     % (n_abnormal_train, train_size, n_abnormal_test, test_size))


    print("保存文件: ../../silicon_data_y/" + name[i])
    np.save("../../silicon_data_y/" + name[i], y_pred_train )