"""
==========================================
SVM
==========================================
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

####### 一些参数 #######
TRAIN = 200
TEST = 200
ABNORMAL_RATE = 0.4
MIU = 0
SIGMA = 0.1
MIU_ABNORMAL = 1
SIGMA_ABNORMAL = 0.1

XMIN = -1.5
YMIN = -1.5
XMAX = 1.5
YMAX = 1.5
# 绘图的点大小
s = 5

def generate_X(data_size, abnormal_rate, miu, sigma, miu_ab, sigma_ab):
    abnormal_size = int(data_size * abnormal_rate)
    normal_size = data_size - abnormal_size

    normal_set = np.random.normal(miu, sigma, (normal_size, 2))
    abnormal_set = np.random.normal(miu_ab, sigma_ab, (abnormal_size, 2))
    return np.concatenate((normal_set, abnormal_set),axis=0), normal_set, abnormal_set

def generate_Y(data_size, abnormal_rate):
    abnormal_size = int(data_size * abnormal_rate)
    normal_size = data_size - abnormal_size

    normal_set = np.zeros(normal_size)
    abnormal_set = np.ones(abnormal_size)
    return np.concatenate((normal_set, abnormal_set),axis=0)

####### 训练集 #######  Generate train data
X_train, X_train_normal, X_train_abnormal = generate_X(TRAIN, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
Y_train = generate_Y(TRAIN, ABNORMAL_RATE)

print("训练集总样本点：", TRAIN, "，训练集异常样本点：", Y_train[Y_train == 1].size, "，占比：", round(Y_train[Y_train == 1].size / TRAIN, 2))

####### 测试集 ####### Generate some novel observations，这里注意和训练集是独立同分布的
X_test, X_test_normal, X_test_abnormal = generate_X(TEST, ABNORMAL_RATE, MIU, SIGMA, MIU_ABNORMAL, SIGMA_ABNORMAL)
Y_test = generate_Y(TEST, ABNORMAL_RATE)

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
