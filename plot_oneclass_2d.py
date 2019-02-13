"""
==========================================
2-D SVM
author: Wan Bingqi
==========================================
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

####### 一些参数 #######
train_size = 10000
test_size = 1024
miu = 0
sigma = 0.1

####### 训练集 #######  Generate train data
X_train = np.random.normal(miu, sigma, (train_size, 2))
Y_train = np.zeros(train_size)

for i in range(0, train_size):
    # if abnormal, Y_train = 1
    if (X_train[i][0] * X_train[i][0] + X_train[i][1] * X_train[i][1] > (sigma * 2) * (sigma * 2)):
        Y_train[i] = 1
print("总样本点：", train_size, "，异常样本点：", Y_train[Y_train == 1].size)

####### 测试集 #######
# Generate some regular novel observations
X_test = np.random.normal(miu, sigma, (test_size, 2))
Y_test = np.zeros(test_size)

for i in range(0, test_size):
    # if abnormal, Y_test = 1
    if (X_test[i][0] * X_test[i][0] + X_test[i][1] * X_test[i][1] > (sigma * 2) * (sigma * 2)):
        Y_test[i] = 1

# Generate some abnormal novel observations
X_outliers = np.random.normal(miu, sigma, (test_size, 2))
for i in range(650,800):
    X_outliers[i][:] = X_outliers[i][:] + 0.1


####### svm #######
# fit the model
# clf = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.1)
clf = svm.SVC(gamma="auto")
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == 1].size
n_error_test = y_pred_test[y_pred_test == 1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# 统计训练集上的表现
cnt = 0
for i in range(0, test_size):
    # if the prediction is same to groundtruth, increase the counter
    if (Y_test[i] == y_pred_test[i]):
        cnt += 1


####### 画图 #######
# 网格的粒度是第三个参数
xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("OneClassSVM by Diana")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')

# X, Y的显示上下界在这里修改
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=8))
plt.xlabel(
    "error train: %d/%d ; errors novel regular: %d/%d ; "
    "errors novel abnormal: %d/%d"
    % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))

plt.text(
    0.5, -0.8,
    "test mIoU: %.2f"
    % (cnt / test_size))
    
plt.show()
