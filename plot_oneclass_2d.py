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

train_size = 1000
test_size = 1024

### 网格的粒度是第三个参数
xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
# Generate train data
X_train = np.random.normal(0,0.1,(train_size,2))
for i in range(0,train_size):
    X_train[i][1] = 0
# Generate some regular novel observations
a = np.random.normal(0,0.1,(test_size,2))
for i in range(0,test_size):
    a[i][1] = 0
for i in range(650,800):
    a[i][0] = a[i][0] + 0.1
    # a[i][:] = a[i][:] + 0.1
X_test = np.r_[a[0:649], a[800:1024]]
# Generate some abnormal novel observations
X_outliers = a[650:800]


# fit the model
clf = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# print (n_error_train, n_error_test, n_error_outliers);

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

### X, Y的显示上下界
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
plt.show()
