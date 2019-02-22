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
train_size = 10000
test_size = 10000
miu = 0
sigma = 0.1

raw_data = np.loadtxt("../data/2327_20170131-03-zs.csv",delimiter=",",skiprows=1)

####### 训练集 #######
# start = round(random.random() * raw_data.shape[0])
start = round(0.2 * raw_data.shape[0])
print(start)
X_train = raw_data[start:start+train_size,:]
print(X_train.shape)


####### 测试集 #######
# start = round(random.random() * raw_data.shape[0])
start = round(0.3 * raw_data.shape[0])
print(start)
X_test = raw_data[start:start+test_size,:]
print(X_test.shape)

# Generate some abnormal novel observations
X_outliers = np.random.normal(miu, sigma, (test_size, 42))

####### svdd #######
# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print ("error train: %d/%d ; errors test regular: %d/%d ; "
    "errors test abnormal: %d/%d"
    % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))

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