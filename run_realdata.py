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
from utils.utils import f1calc
import numpy as np
import time

i = 0
LEVEL = 6
do_wavelet = True
do_calibration = True
do_split = True
do_stats = True
do_svm = True
do_pca = True

filename = '2437_20161108-07-zs/1'

original_path = '../data_raw/'
wavelet_path = '../data/'
ground_truth_path = '../data_gt/'

def stats(ground_truth_path, filename):
    Y_train = np.load(ground_truth_path + filename + '_train.npy')
    Y_test = np.load(ground_truth_path + filename + '_test.npy')
    y_train_baseline = np.load(ground_truth_path + filename + '_train_ewma.npy')
    y_test_baseline = np.load(ground_truth_path + filename + '_test_ewma.npy')
    # 统计精度，召回率
    recall, precision, F1, acc = f1calc(Y_train, y_train_baseline)
    print ("EWMA train set recall: ", recall,"; precision:", precision, "F1-score:", F1, "acc:", acc)
    recallTe, precisionTe, F1Te, accTe = f1calc(Y_test, y_test_baseline)
    print ("EWMA test set recall: ", recallTe,"; precision:", precisionTe, "F1-score:", F1Te, "acc:", accTe)


if (do_wavelet):
    # 执行小波变换
    time_start = time.time()
    process_wavelet(original_path, wavelet_path, filename, LEVEL)
    time_end = time.time()
    print("小波变换用时:",time_end - time_start,"秒")

if (do_calibration):
    # 数据标注
    calibrate(wavelet_path, ground_truth_path, filename)

if (do_split):
    # 将数据按奇偶分别分为训练集与测试集
    split(wavelet_path, ground_truth_path, filename)

if (do_stats):
    stats(ground_truth_path, filename)

if (do_svm):
    # 执行SVM
    time_start = time.time()
    doSvm(wavelet_path + filename + '_train.npy', wavelet_path + filename + '_test.npy', ground_truth_path + filename + '_train.npy', ground_truth_path + filename + '_test.npy')
    time_end = time.time()
    print("svm用时:",time_end - time_start,"秒")

if (do_pca):
    # 执行pca，画图
    X = np.load(wavelet_path + filename + '_test.npy')
    y = np.load(ground_truth_path + filename + '_test.npy')
    plot_pca(X, y)