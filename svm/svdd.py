"""
==========================================
SVDD
==========================================
"""

import numpy as np
from sklearn import svm

def svdd(inputName, outputName):

    X_train = np.load(inputName)

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

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    # y_pred_test = clf.predict(X_test)
    n_abnormal_train = y_pred_train[y_pred_train == -1].size
    # n_abnormal_test = y_pred_test[y_pred_test == -1].size

    print ("abnormal: %d/%d ;"
        % (n_abnormal_train, X_train.size))

    # print ("abnormal train: %d/%d ; abnormal test: %d/%d ; "
    #     % (n_abnormal_train, train_size, n_abnormal_test, test_size))


    print("保存文件: " + outputName)
    np.save(outputName, y_pred_train )