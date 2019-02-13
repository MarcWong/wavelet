"""
==========================================
2-D SVM
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
p = 2

####### 训练集 #######  Generate train data
X_train = np.random.normal(miu, sigma, (train_size, 2))
Y_train = np.zeros(train_size)

for i in range(0, train_size):
    # if abnormal, Y_train = 1
    if (X_train[i][0] * X_train[i][0] + X_train[i][1] * X_train[i][1] > (sigma * p) * (sigma * p)):
        Y_train[i] = 1
print("训练集总样本点：", train_size, "，训练集异常样本点：", Y_train[Y_train == 1].size, "，占比：", round(Y_train[Y_train == 1].size / train_size, 2))

####### 测试集 #######
# Generate some regular novel observations
X_test = np.random.normal(miu, sigma, (test_size, 2))
Y_test = np.zeros(test_size)

for i in range(0, test_size):
    # if abnormal, Y_test = 1
    if (X_test[i][0] * X_test[i][0] + X_test[i][1] * X_test[i][1] > (sigma * p) * (sigma * p)):
        Y_test[i] = 1
print("正常训练集总样本点：", test_size, "，训练集负样本点：", Y_test[Y_test == 1].size, "，占比：", round(Y_test[Y_test == 1].size / test_size, 2))

# Generate some abnormal novel observations
X_outliers = np.random.normal(miu, sigma, (test_size, 2))
Y_outliers = np.zeros(test_size)
for i in range(0,test_size):
    X_outliers[i][:] = X_outliers[i][:] + 0.2
    # if abnormal, Y_outliers = 1
    if (X_outliers[i][0] * X_outliers[i][0] + X_outliers[i][1] * X_outliers[i][1] > (sigma * p) * (sigma * p)):
        Y_outliers[i] = 1
print("异常训练集总样本点：", test_size, "，训练集负样本点：", Y_outliers[Y_outliers == 1].size, "，占比：", round(Y_outliers[Y_outliers == 1].size / test_size, 2))

####### svm #######
# fit the model
clf = svm.SVC(gamma="auto")
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == 1].size
n_error_test = y_pred_test[y_pred_test == 1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# 统计正常训练集上的表现
cntRegular = 0
for i in range(0, test_size):
    # 判断预测值与真值是否一致
    if (Y_test[i] == y_pred_test[i]):
        cntRegular += 1
# 统计异常训练集上的表现
cntIrregular = 0
for i in range(0, test_size):
    if (Y_outliers[i] == y_pred_outliers[i]):
        cntIrregular += 1


####### 画图 #######
# 网格的粒度是第三个参数
xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("2-D SVM")
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
           ["learned frontier", "train set",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=8))
plt.xlabel(
    "error train: %d/%d ; errors test regular: %d/%d ; "
    "errors test abnormal: %d/%d"
    % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))

plt.text(
    -0.1, -0.8,
    "normal test set correctness: %.2f"
    % (cntRegular / test_size))
plt.text(
    -0.1, -0.9,
    "abnormal test set correctness: %.2f"
    % (cntIrregular / test_size))

plt.show()
