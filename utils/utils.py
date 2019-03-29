import numpy as np

def f1calc(gt, pred):
    print(gt.shape)
    print(pred.shape)
    ra = gt.shape[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, ra):
        # 判断预测值与真值是否一致
        if (gt[i] == pred[i] == 0):
            TP += 1
        elif (gt[i] == pred[i] == 1):
            TN += 1
        elif (gt[i] != pred[i] and gt[i] == 1):
            FP += 1
        else:
            FN += 1
    print ('TP:', TP)
    print ('TN:', TN)
    print ('FP:', FP)
    print ('FN:', FN)
    if (TP+FP == 0):
        return TP/ (TP+FN), 0, 2*TP/(2*TP + FP + FN), (TP + TN) / (TP + TN + FP + FN)
    return TP/ (TP+FN), TP/ (TP+FP), 2*TP/(2*TP + FP + FN), (TP + TN) / (TP + TN + FP + FN)

def stats(Y, ub, lb):
    N = Y.shape[0];
    result = np.ones(N , dtype=int)
    for i in range(0, N):
        if Y[i] < ub[i] and Y[i] > lb[i]:
            result[i] = 0
    return result


def statsC(Y, ub, lb, N):
    result = np.ones(N, dtype=int)
    sample_rate = N / Y.shape[0]
    for i in range(0, N):
        if Y[int(i * sample_rate)] < ub and Y[int(i * sample_rate)] > lb:
            result[i] = 0
    return result