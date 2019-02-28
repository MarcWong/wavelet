"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

OUTLIER_LABEL = 1
NORMAL_LABEL = -1

OUTLIER_DEVIATION = 0.1

TEST_SHIFT = 0.5

train_size = 4000
test_size = 1024
miu = 0
sigma = 0.1

#   随机生成的数据要make sense，指定类别的标准必须是相同的（独立同分布原则），所以单独作为一个函数
def generate_label(X):
    data_size = X.shape[0]
    Y = np.zeros(data_size)
    for i in range(0, data_size):
        # if abnormal, Y_train = 1
        if X[i][0] ** 2 + X[i][1] > OUTLIER_DEVIATION:  
            # TODO 这里改了。改的原因是，一阶的判断标准应该和均匀分布搭配，二阶以上的的标准应该和正态分布搭配。如果要以其他标准来判断outlier，就改这里
            # 挑一部分点加shift作为outliers
            Y[i] = OUTLIER_LABEL
        else:
            Y[i] = NORMAL_LABEL
    print('generated {} normal points and {} outliers'.format(np.sum(Y == NORMAL_LABEL), np.sum(Y == OUTLIER_LABEL)))
    return Y


# Generate train data
X_train = np.random.normal(miu, sigma, (train_size, 2))
Y_train = generate_label(X_train)
x_outlier_in_train = X_train[Y_train == OUTLIER_LABEL]
x_normal_in_train = X_train[Y_train == NORMAL_LABEL]


print('number of outliers in train set:', Y_train[Y_train == OUTLIER_LABEL].size)
#for i in range(0,train_size):
  #  X_train[i][1] = 0
# Generate some regular novel observations
X_test_all = np.random.normal(miu, sigma, (test_size, 2))
#for i in range(0,test_size):
   # a[i][1] = 0
for i in range(650,800):
    #a[i][0] = a[i][0] + 0.1
    X_test_all[i][:] = X_test_all[i][:] + TEST_SHIFT

#之前下面三行错了，并不能说加上0.1的就是离群点
# X_test = np.r_[a[0:649], a[800:1024]]
# # Generate some abnormal novel observations
# X_test_outliers = a[650:800]

Y_test_all = generate_label(X_test_all)
X_test_outliers = X_test_all[Y_test_all == OUTLIER_LABEL]
X_test_normal = X_test_all[Y_test_all == NORMAL_LABEL]

# fit the model
# clf = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.1)
clf = svm.SVC()
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
n_pred_outlier_in_train = y_pred_train[y_pred_train == OUTLIER_LABEL].size  # 在训练数据中判断出来的outlier
print('predicted {} outliers in the training set'.format(n_pred_outlier_in_train))

y_pred_test = clf.predict(X_test_all)
x_pred_outlier_in_test = X_test_all[y_pred_test == OUTLIER_LABEL]
x_pred_normal_in_test = X_test_all[y_pred_test == NORMAL_LABEL]
n_pred_outlier_in_test = x_pred_outlier_in_test.shape[0] # 在测试数据中判断出来的outlier有这么多
n_pred_normal_in_test = x_pred_normal_in_test.shape[0] # 在测试数据中判断出来的normal有这么多
print('predicted {} normal points and {} outliers in the test set'.format(n_pred_normal_in_test, n_pred_outlier_in_test))


#   画decision boundary
xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("OneClassSVM by Diana")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

#   把判断出来的正常点和离群点画出来
s = 1
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(x_pred_normal_in_test[:, 0], x_pred_normal_in_test[:, 1], c='blueviolet', s=s)
c = plt.scatter(x_pred_outlier_in_test[:, 0], x_pred_outlier_in_test[:, 1], c='gold', s=s)

train_normal_points = plt.scatter(x_normal_in_train[:, 0], x_normal_in_train[:, 1], s=s)
train_outlier_points = plt.scatter(x_outlier_in_train[:, 0], x_outlier_in_train[:, 1], s=s)

plt.axis('tight')
plt.xlim((-1, 1))   ### X, Y的显示上下界
plt.ylim((-1, 1))
plt.legend([a.collections[0], b2, c, train_normal_points, train_outlier_points],
           ["learned frontier",
            "test normal", "test outliers", 'train normal', 'train outliers'],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=8))

#   计算正确率。正确的判断包括两部分：把outlier判断成outlier，把normal判断成normal
n_outlier_as_outlier = 0
n_normal_as_normal = 0
for i in range(test_size):
    if y_pred_test[i] == OUTLIER_LABEL and Y_test_all[i] == OUTLIER_LABEL:
        n_outlier_as_outlier += 1
    elif y_pred_test[i] == NORMAL_LABEL and Y_test_all[i] == NORMAL_LABEL:
        n_normal_as_normal += 1
accuracy = (n_outlier_as_outlier+n_normal_as_normal) / test_size
msg = 'among {} test outliers, discovered {} ;\n among {} test normal points, predicted {} as normal ;\n the accuracy is {}/{}={}'.format(np.sum(Y_test_all == OUTLIER_LABEL),
    n_outlier_as_outlier, np.sum(Y_test_all == NORMAL_LABEL), n_normal_as_normal, n_outlier_as_outlier+n_normal_as_normal, test_size, accuracy)
plt.xlabel(msg)

# plt.xlabel(
#     "error train: %d/%d ; errors novel regular: %d/%d ; "
#     "errors novel abnormal: %d/%d"
#     % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))


# plt.xlabel(
#     "error train: %d/%d ; errors novel regular: %d/%d ; "
#     "errors novel abnormal: %d/%d"
#     % (n_error_train, train_size, n_error_test, test_size, n_error_outliers, test_size))
plt.show()



# print (n_error_train, n_error_test, n_error_outliers);

# plot the line, the points, and the nearest vectors to the plane




