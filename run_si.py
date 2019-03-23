# -*- coding: utf-8 -*-
"""
==========================================
silicon数据脚本，目前是拆分成了三个：
先跑process_wavelet.py
再跑svdd.py
再跑run_si.py
==========================================
"""
print(__doc__)
from pca.plot_pca_wavelet import plot_pca
import numpy as np

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

X = np.load('../silicon_data/' + name[0])
y = np.load('../silicon_data_y/' + name[0])

plot_pca(X, y)