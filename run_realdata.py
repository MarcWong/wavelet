# -*- coding: utf-8 -*-
"""
==========================================
真实数据脚本
==========================================
"""
print(__doc__)
from pca.plot_pca_wavelet import plot_pca
from process_wavelet import process_wavelet
from svm.svm import doSvm
from utils.calibrate import calibrate
from utils.split import split
import numpy as np

i = 0
LEVEL = 6
do_wavelet = True
do_calibration = True
do_split = True
do_svm = True
do_pca = True

filename = '2437_20161108-07-zs/1'

original_path = '../data_raw/'
wavelet_path = '../data/'
ground_truth_path = '../data_gt/'


if (do_wavelet):
    # 执行小波变换
    process_wavelet(original_path, wavelet_path, filename, LEVEL)

if (do_calibration):
    # 数据标注
    calibrate(wavelet_path, ground_truth_path, filename)

if (do_split):
    # 将数据按奇偶分别分为训练集与测试集
    split(wavelet_path, ground_truth_path, filename)

if (do_svm):
    # 执行SVM
    doSvm(wavelet_path + filename + '_train.npy', wavelet_path + filename + '_test.npy', ground_truth_path + filename + '_train.npy', ground_truth_path + filename + '_test.npy')

if (do_pca):
    # 执行pca，画图
    X = np.load(wavelet_path + filename + '_test.npy')
    y = np.load(ground_truth_path + filename + '_test.npy')
    plot_pca(X, y)