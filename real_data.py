import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

##### Set parameters

N = 20000
tau = 12000
n = 128
J = int(np.log2(n)) - 1
dif = int(np.log2(n)) - J
mu = 0.0
sd = 1
shift = 0.5
drift = 0.001
wavelet = 'haar'
thres_level = 1

##### VisuShrink

def hard(x, t):
    return ((abs(x) - t) > 0) * x

def soft(x, t):
    return np.sign(x) * np.maximum(abs(x) - t, np.zeros(len(x)))

def VisuShrink(c, n, j0, J):
    d = []
    for j in range(j0):
        d.append(c[j])
    for j in range(j0, J + 1):
        x = c[j]
        sigmaj = np.median(abs(x - np.median(x))) / 0.6745
        tj = sigmaj * np.sqrt(2 * np.log(n))
        d.append(soft(x, tj))
    return d

def UCL(i, mu0, L, sigma, lam):
    return mu0 + L * sigma *np.sqrt(lam/(2 - lam) * (1 - (1 - lam)**(2*i)))
def LCL(i, mu0, L, sigma, lam):
    return mu0 - L * sigma *np.sqrt(lam/(2 - lam) * (1 - (1 - lam)**(2*i)))


def stats(Y, ub, lb):
    # N = Y.shape[0];
    N = len(Y)
    result = np.ones(N , dtype=int)
    for i in range(0, N):
        if Y[i] < ub[i] and Y[i] > lb[i]:
            result[i] = 0
    return result

def calibrate(X):
    y_gt = np.zeros(len(X), dtype=int);
    for i in range(11400, 11500, 2):
        y_gt[i] = 1;
    for i in range(21400, 21500, 2):
        y_gt[i] = 1;
    return y_gt


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


##### Generate data

data = pd.read_csv('coef_data/2327_20170131-03-zs.csv')
t1_4 = np.array(data[["t27", "t28", "t29", "t30"]])
m = t1_4.shape[1]
N = t1_4.shape[0]
Y = t1_4[:, 2]

#plt.figure(figsize=(8, 5.0))
#plt.plot(t1_4[:,0], color = "black")
#plt.title("Streaming Data")
#plt.show()

#for j in range(0,m):
#    t1_4[t1_4[:,j] == -50, j] = np.mean(t1_4[:,j])

var_coef1 = pd.read_csv('coef_data/coef29.txt')
var_coef1 = np.array(var_coef1.iloc[:,1])

plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(15, 2.4*4))
plt.subplot(4,1,1)
plt.ylabel("t27")
plt.plot(t1_4[:,0], color = "black")
plt.subplot(4,1,2)
plt.ylabel("t28")
plt.plot(t1_4[:,1], color = "black")
plt.subplot(4,1,3)
plt.ylabel("t29")
plt.plot(t1_4[:,2], color = "black")
plt.subplot(4,1,4)
plt.ylabel("t30")
plt.plot(t1_4[:,3], color = "black")
# plt.show()
plt.savefig("2327_20170131-03-zs_29-32.png")


## DIRECT EWMA PREDICTION
lam = 0.5
Y_origin_ewma = [Y[0]]
for i in range(1, N):
    Y_origin_ewma.append(lam * Y[i] + (1 - lam) * Y_origin_ewma[i-1])
resid = Y - Y_origin_ewma
plt.figure(figsize=(14,9))
plt.plot(resid)
plt.plot(Y)


##### Use package

Y_pred = var_coef1[m] + np.dot(t1_4[0:(n-1),:], var_coef1[:m])
Y_pred = np.append(Y[0], Y_pred)
y = Y[0:n] - Y_pred
c = pywt.wavedec(y, wavelet, level = J)
c_visu = VisuShrink(c, n, thres_level, J)
y_denoise = pywt.waverec(c_visu, wavelet)
Y_denoise = y_denoise.tolist()

time_start = time.time()

for i in range(n, N):
    x = Y[i]
    if x == -50:
        t1_4[i,:] = t1_4[(i-1),:]
        y = np.append(y, 0) ## COPY THE LAST POINT
    else:
        pred = var_coef1[m] + np.dot(t1_4[(i-1),:], var_coef1[:m])
        Y_pred = np.append(Y_pred, pred)
        y = np.append(y, x - pred)
    profile = y[(i - n + 1):(i + 1)]
    c = pywt.wavedec(profile, wavelet, level = J)
    c_visu = VisuShrink(c, n, thres_level, J)
    y_denoise = pywt.waverec(c_visu, wavelet)
    Y_denoise.append(y_denoise[n - 1])

time_end = time.time()
print('time = ', time_end - time_start)

#denoised = Y[Y - Y_denoise != 0]

lam = 0.5
Y_ewma = [Y_denoise[0]]
for i in range(1, len(Y_denoise)):
    Y_ewma.append(lam * Y_denoise[i] + (1 - lam) * Y_ewma[i - 1])

sigma = np.std(Y_denoise)
ucl = np.zeros(N)
for i in range(N):
    ucl[i] = UCL(i+1, 0, 5, sigma, lam)
lcl = np.zeros(N)
for i in range(N):
    lcl[i] = LCL(i+1, 0, 8, sigma, lam)

plt.plot(Y)
# plt.show()
plt.plot(y)
# plt.show()
plt.plot(Y_denoise)
#plt.ylim((-2, 2))
# plt.show()

plt.plot(Y_ewma)
plt.plot(ucl)
plt.plot(lcl)
# plt.show()

outlier_u = np.where(Y_ewma > ucl)[0]
outlier_l = np.where(Y_ewma < lcl)[0]


#################### MONITOR WAVELET COEFFICIENTS
Y = y

coef = []
for j in range(J + 1):
    coef.append(np.array([]))
coef_denoise = []
for j in range(J + 1):
    coef_denoise.append(np.array([]))

y = Y[0:n]
c = pywt.wavedec(y, wavelet, level = J)
c_visu = VisuShrink(c, n, thres_level, J)
y_denoise = pywt.waverec(c_visu, wavelet)
Y_denoise = y_denoise.tolist()

for i in range(n, N):
    x = Y[i]
    y = np.append(y, x)
    profile = y[(i - n + 1):(i + 1)]
    c = pywt.wavedec(profile, wavelet, level = J)
    c_visu = VisuShrink(c, n, thres_level, J)
    s = 0
    for j in range(J + 1):
        coef[j] = np.append(coef[j], c[j][-1])
        coef_denoise[j] = np.append(coef_denoise[j], c_visu[j][-1])
    y_denoise = pywt.waverec(c_visu, wavelet)
    Y_denoise.append(y_denoise[n - 1])

lam = 0.3
Lu_wrewma = 2.5
Ll_wrewma = 5
sd_wrewma = 0.275

Y_ewma = [Y_denoise[0]]
for i in range(1, N):
    Y_ewma.append(lam * Y_denoise[i] + (1 - lam) * Y_ewma[i - 1])

ucl_wrewma = np.zeros(N)
for i in range(N):
    ucl_wrewma[i] = UCL(i+1, mu, Lu_wrewma, sd_wrewma, lam)
lcl_wrewma = np.zeros(N)
for i in range(N):
    lcl_wrewma[i] = LCL(i+1, mu, Ll_wrewma, sd_wrewma, lam)

plt.plot(Y_ewma)
plt.plot(ucl_wrewma)
plt.plot(lcl_wrewma)
# plt.show()



plt.subplots(nrows=J+4, ncols=1, sharex=True, sharey=False, figsize=(15, 2.3*(J+4)))
plt.subplot((J+3), 1, 1)
plt.ylabel("Original")
plt.plot(t1_4[:, 2], color = "black")
plt.subplot((J+3), 1, 2)
plt.ylabel("Residuals")
plt.plot(Y, color = "black")

lam = 0.6
Lu = [4.8,50,20,20,20,20,20]
Ll = [4.8,60,20,20,20,20,20]

outlier_up = []
outlier_low = []
for j in range(J + 1):
    sd_coef = np.std(coef_denoise[j][30000:60000])
    coef_len = len(coef_denoise[j])
#    sd = t_list[j] / np.sqrt(2 * np.log(n))
    ucl = np.zeros(N)
    for i in range(N):
        ucl[i] = UCL(i+1, mu, Lu[j], sd_coef, lam)
    lcl = np.zeros(N)
    for i in range(N):
        lcl[i] = LCL(i+1, mu, Ll[j], sd_coef, lam)
    coef_ewma = np.zeros(n).tolist()
    for i in range(n, N):
        coef_ewma.append(lam * coef_denoise[j][i-n] + (1 - lam) * coef_ewma[i-1])
    plt.subplot((J+3), 1, j+3)
    if j == 0:
        plt.ylabel("D")
        result = stats(coef_ewma, ucl, lcl)
        Y_gt = calibrate(coef_ewma)
        recall, precision, F1, acc = f1calc(Y_gt, result)
        print ("EWMA train set recall: ", recall,"; precision:", precision, "F1-score:", F1, "acc:", acc)
        
    else:
        plt.ylabel("A" + str(J-j+1))
    plt.plot(coef_ewma, color = "black")
    plt.plot(ucl)
    plt.plot(lcl)
    outlier_up.append(np.where(coef_ewma > ucl)[0])
    outlier_low.append(np.where(coef_ewma < lcl)[0])
outlier_wdewma = np.union1d(np.concatenate(np.array(outlier_up)),
                            np.concatenate(np.array(outlier_low)))
# plt.show()




#################### MEDIAN

data = []
ucl_list = []
lcl_list = []

coef = pd.read_csv('coef_data/A3.txt', sep = '\t', header = None)
coef = np.array(coef)
sd = np.std(coef)
coef_len = len(coef)
ucl = np.zeros(coef_len)
for i in range(coef_len):
    ucl[i] = UCL(i+1, mu, 3, sd, lam)
lcl = np.zeros(coef_len)
for i in range(coef_len):
    lcl[i] = LCL(i+1, mu, 3, sd, lam)
coef_ewma = [coef[0]]
for i in range(1, coef_len):
    coef_ewma.append(lam * coef[i] + (1 - lam) * coef[i - 1])
data.append(coef_ewma)
ucl_list.append(ucl)
lcl_list.append(lcl)

plt.plot(coef_ewma, color = "black")
plt.plot(ucl)
plt.plot(lcl)
# plt.show()

data = [coef_ewma]
data.append(coef_ewma)
ucl

plt.figure(figsize=(14,14))
plt.subplot(411)
plt.plot(data[0], color = "black")
plt.plot(ucl_list[0])
plt.plot(lcl_list[0])
plt.subplot(412)
plt.plot(data[1], color = "black")
plt.plot(ucl_list[1])
plt.plot(lcl_list[1])
plt.subplot(413)
plt.plot(data[2], color = "black")
plt.plot(ucl_list[2])
plt.plot(lcl_list[2])
plt.subplot(414)
plt.plot(data[3], color = "black")
plt.plot(ucl_list[3])
plt.plot(lcl_list[3])
# plt.show()






