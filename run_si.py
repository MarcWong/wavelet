# -*- coding: utf-8 -*-
"""
==========================================
silicon数据脚本
==========================================
"""
print(__doc__)
from pca.plot_pca_wavelet import plot_pca
from process_wavelet import process_wavelet
from svm.svdd import svdd
import numpy as np

i = 0
LEVEL = 6
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


# 执行小波变换
input_wavelet = '../silicon/' + name[i]
output_wavelet = '../silicon_data/' + name[i]
process_wavelet(input_wavelet, output_wavelet, LEVEL)


# 执行SVDD
input_SVDD = '../silicon_data/' + name[i]
output_SVDD = '../silicon_data_y/' + name[i]
svdd(input_SVDD, output_SVDD)

# 执行pca，画图
# X = np.load('../silicon_data/' + name[i])
# y = np.load('../silicon_data_y/' + name[i])
# plot_pca(X, y)