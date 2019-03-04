def f1calc(gt, pred):
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
    return TP/ (TP+FN), TP/ (TP+FP), 2*TP/(2*TP + FP + FN)