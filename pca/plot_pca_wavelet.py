#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
==========================================
pipeline3-pca
第三步，pca降维到3维，在三维重建中画图
==========================================
"""

# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

def plot_pca():
    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    X = np.load('../data/simulate/X_train.npy')
    y = np.load('../data/simulate/Y_train.npy')
    # X_test = np.load('../data/simulate/X_test.npy')
    # Y_test = np.load('../data/simulate/Y_test.npy')


    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [('Normal', 0), ('Abnormal', 1)]:
        ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean() + 1.5,
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [0, 1]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.bwr,
            edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
