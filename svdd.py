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
data_size = 1024


####### 训练集 #######
X_train = np.load('../data/2327_20170131-03-zs/1.npy').reshape(-1, 1)
print(X_train.shape)


####### 测试集 #######
X_test = np.load('../data/2678_20161209-06-zs/1.npy').reshape(-1, 1)
print(X_test.shape)


####### svdd #######
# fit the model

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.fit_predict(X_train)
y_pred_test = clf.predict(X_test)
n_abnormal_train = y_pred_train[y_pred_train == -1].size
n_abnormal_test = y_pred_test[y_pred_test == -1].size

print ("abnormal train: %d/%d ; abnormal test: %d/%d ; "
    % (n_abnormal_train, 1024, n_abnormal_test, 1024))

####### 画图 #######
# 网格的粒度是第三个参数
# xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-1, 1))
# plt.ylim((-1, 1))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/%d ; errors test regular: %d/%d ; "
#     "errors test abnormal: %d/%d"
#     % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))
# plt.show()