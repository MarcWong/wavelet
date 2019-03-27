import pywt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels.api as sm
import statsmodels.tsa as tsa
from utils.utils import stats

#################### Set parameters

# N = 5000
tau = 3000
mu = 0.0
sd = 1
wavelet_method = 'haar'
thres_level = 1
shift = 0.5
broken_range = 500
broken = shift/broken_range
para_range = 500
parabolic = shift / para_range**2
spike = shift * 8
spike_range = 5

#################### VisuShrink

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


def wavelet(Y, J):
    N = Y.size;
    n = 2**(J+1)
    dif = int(np.log2(n)) - J
    #################### Generate data

    ## SHIFT
    # y1 = np.random.normal(mu, sd, tau)
    # y2 = np.random.normal(mu + shift, sd, N - tau)
    # Y = np.append(y1, y2)

    ## DRIFT (BROKEN LINE)
    ##np.random.seed(18)
    ##Y = np.random.normal(loc = mu, scale = sd, size = tau)
    ##for i in range(broken_range):
    ##    Y = np.append(Y, np.random.normal(loc = mu + i*broken, scale = sd, size = 1))
    ##y2 = np.random.normal(loc = mu + shift, scale = sd, size = N - tau - broken_range)
    ##Y = np.append(Y, y2)

    ## DRIFT (PARABOLIC)
    ##np.random.seed(18)
    ##Y = np.random.normal(loc = mu, scale = sd, size = tau)
    ##for i in range(para_range):
    ## Y = np.append(Y, np.random.normal(loc = mu + i**2 * parabolic, scale = sd, size = 1))
    ##y2 = np.random.normal(loc = mu + shift, scale = sd, size = N - tau - para_range)
    ##Y = np.append(Y, y2)

    ## SPIKE
    ##np.random.seed(18)
    ##Y = np.random.normal(loc = mu, scale = sd, size = N)
    ##Y[(tau+1) : (tau+spike_range)] = Y[(tau+1) : (tau+spike_range)] + spike

    ## VARIANCE CHANGE





    ## Compute threshold
    #t_list = []
    #for j in range(30):
    #    y = Y[(j*n):((j+1)*n)]
    #    c = pywt.wavedec(y, wavelet_method, level = J)
    #    t = thres(c, n, J)
    #    t_list.append(t)
    #t_list = np.array(t_list)
    #t_list = np.mean(t_list, 0)




    ################### COMPUTE 3 METHODS

    ## WREWMA AND COEFFICIENTS
    coef = []
    for j in range(J + 1):
        coef.append(np.array([]))
    coef_denoise = []
    for j in range(J + 1):
        coef_denoise.append(np.array([]))

    y = Y[0:n]
    c = pywt.wavedec(y, wavelet_method, level = J)
    c_visu = VisuShrink(c, n, thres_level, J)
    y_denoise = pywt.waverec(c_visu, wavelet_method)
    Y_denoise = y_denoise.tolist()

    for i in range(n, N):
        x = Y[i]
        y = np.append(y, x)
        profile = y[(i - n + 1):(i + 1)]
        c = pywt.wavedec(profile, wavelet_method, level = J)
        c_visu = VisuShrink(c, n, thres_level, J)
        s = 0
        for j in range(J + 1):
            coef[j] = np.append(coef[j], c[j][-1])
            coef_denoise[j] = np.append(coef_denoise[j], c_visu[j][-1])
        y_denoise = pywt.waverec(c_visu, wavelet_method)
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

    plt.plot(Y_ewma,color = "black")
    plt.plot(ucl_wrewma, color = "black")
    plt.plot(lcl_wrewma, color = "black")
    plt.show()



    ## BASE
    lam = 0.1
    Lu_base = 3.0
    Ll_base = 5
    sd = 1

    Y_base = [Y[0]]
    for i in range(1, N):
        Y_base.append(lam * Y[i] + (1 - lam) * Y_base[i - 1])

    ucl_base = np.zeros(N)
    for i in range(N):
        ucl_base[i] = UCL(i+1, mu, Lu_base, sd, lam)
    lcl_base = np.zeros(N)
    for i in range(N):
        lcl_base[i] = LCL(i+1, mu, Ll_base, sd, lam)

    # plt.plot(Y_base,color = "black")
    # plt.plot(ucl_base,color = "black")
    # plt.plot(lcl_base,color = "black")
    # plt.show()


    #################### PLOT 3 METHODS

    # plt.subplots(nrows=J+4, ncols=1, sharex=True, sharey=False, figsize=(15, 2.3*(J+4)))
    # plt.subplot((J+4), 1, 1)
    # plt.ylabel("Original")
    # plt.plot(Y, color = "black")

    ## BASE

    # plt.subplot((J+4), 1, 2)
    # plt.ylabel("EWMA")
    # plt.plot(Y_base, color = "black")
    # plt.plot(ucl_base,color = "black")
    # plt.plot(lcl_base,color = "black")

    ## WREWMA
    # plt.subplot((J+4), 1, 3)
    # plt.ylabel("WREWMA")
    # plt.plot(Y_ewma, color = "black")
    # plt.plot(ucl_wrewma,color = "black")
    # plt.plot(lcl_wrewma,color = "black")

    lam = 0.6
    Lu = [3.6,20,25,30,40,40,40,40]
    Ll = [6.0,20,25,30,40,40,40,40]

    outlier_up = []
    outlier_low = []
    for j in range(J + 1):
        sd_coef = np.std(coef_denoise[j][0:tau])
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
        plt.subplot((J+4), 1, j+4)
        if j == 0:
            plt.ylabel("D")
        else:
            plt.ylabel("A" + str(J-j+1))
        plt.plot(coef_ewma, color = "black")
        plt.plot(ucl,color = "blueviolet")
        plt.plot(lcl,color = "coral")

        if j == J:
            result = stats(Y, ucl, lcl, N)

    plt.show()

    return np.array(coef_denoise).T, result

    """

    #################### OUT OF CONTROL EXAMPLE

    plt.subplots(nrows=J+4, ncols=1, sharex=True, sharey=False, figsize=(15, 2.4*4))
    y = np.zeros(N)
    for i in range(tau):
        y[i] = 0
    for i in range(tau, N):
        y[i] = shift
    plt.subplot(4, 1, 1)
    plt.ylabel("Shift")
    plt.plot(y, color = "black")
    for i in range(tau):
        y[i] = 0
    for i in range(tau, tau + broken_range):
        y[i] = broken * (i - tau)
    for i in range(tau + broken_range, N):
        y[i] = shift
    plt.subplot(4, 1, 2)
    plt.ylabel("Drift (broken-line)")
    plt.plot(y, color = "black")
    for i in range(tau):
        y[i] = 0
    for i in range(tau, tau + para_range):
        y[i] = parabolic * (i - tau)**2
    for i in range(tau + para_range, N):
        y[i] = shift
    plt.subplot(4, 1, 3)
    plt.ylabel("Drift (parabolic)")
    plt.plot(y, color = "black")
    y = np.zeros(N)
    for i in range(tau, tau + spike_range):
        y[i] = spike
    plt.subplot(4, 1, 4)
    plt.ylabel("Spike")
    plt.plot(y, color = "black")



    #################### ARIMA

    arparams = np.array([1, -0.60, 0.15])
    maparams = np.array([1, 0.10])
    p0 = len(arparams) - 1
    q0 = len(maparams) - 1

    nobs = 2000
    np.random.seed(2014)
    y = arma_generate_sample(arparams, maparams, nobs, sigma = sd)

    res = sm.tsa.arma_order_select_ic(
            y, max_ar = 4, max_ma = 2, ic='aic', trend='c')
    print(res.aic_min_order)
    (p, q) = res.aic_min_order


    np.random.seed(1000)
    Y = arma_generate_sample(arparams, maparams, tau, sigma = sd)
    model = tsa.arima_model.ARIMA(Y[0:tau], order = (p, 0, q)).fit(
            trend='c', disp = -1)
    phi = model.arparams
    theta = model.maparams
    e = model.resid
    for i in range(tau, N):
        tmp = np.random.normal(loc = mu, scale = sd, size = 1)
        new = arparams[0] * shift + sum(-arparams[1:] * Y[(i-1):(i-p0-1):(-1)]) \
        + sum(maparams[1:] * e[(i-1):(i-q0-1):(-1)]) + tmp
        Y = np.append(Y, new)
        e = np.append(e, tmp)
    plt.plot(Y)


    y_star = [Y[0]]
    fit = [Y[0]]
    y = [0]
    for l in range(1, N):
        y_star.append(Y[l-1])
        pred = sum(phi * y_star[(l-2):(l-p-2):(-1)]) \
        + sum(theta * y[(l-2):(l-q-2):(-1)])
        fit.append(pred)
        y.append(y_star[l-1] - pred)
    plt.plot(y)


    plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(18, 4.0*2))
    plt.subplot(2,1,1)
    plt.ylabel("Original")
    plt.plot(Y, color = "black")
    plt.subplot(2,1,2)
    plt.ylabel("Residuals")
    plt.plot(y, color = "black")



    #################### SARIMA

    shift = 1.5
    arparams = np.array([1])
    maparams = np.array([1, 0.30])
    p0 = len(arparams) - 1
    q0 = len(maparams) - 1
    ARparams = np.array([1, -0.55, -0.45])
    MAparams = np.array([1, -0.50])
    P0 = len(ARparams) - 1
    Q0 = len(MAparams) - 1
    sin_scale = 12
    s = 2*sin_scale

    phi = np.array([0])
    theta = np.array([0.2769])
    Phi = np.array([1])
    Theta = np.array([-0.7296])
    p = len(phi)
    q = len(theta)
    P = len(Phi)
    Q = len(Theta)

    np.random.seed(1987)

    y = np.array([])
    epsilon = []
    for i in range(4*s):
        e = np.random.normal(loc = mu, scale = sd, size = 1)
        epsilon = np.append(epsilon, e)
        y = np.append(y, 30*np.sin(i * np.pi/sin_scale) + e)
    for i in range(4*s, tau):
        e = np.random.normal(loc = mu, scale = sd, size = 1)
        epsilon = np.append(epsilon, e)
        yi = mu - sum(arparams[1:]*(y[(i-1):(i-p0-1):(-1)] - mu)) \
        - ARparams[1] * sum(arparams*(y[(i-s):(i-s-p0-1):(-1)] - mu)) \
        - ARparams[2] * sum(arparams*(y[(i-2*s):(i-2*s-p0-1):(-1)] - mu)) \
        + sum(maparams*epsilon[i:(i-q0-1):(-1)]) \
        + MAparams[1] * sum(maparams*epsilon[(i-s):(i-s-q0-1):(-1)])
        y = np.append(y, yi)
    #for i in range(tau, (tau + 2*s)):
    #    e = np.random.normal(loc = mu, scale = sd, size = 1)
    #    epsilon = np.append(epsilon, e)
    #    yi = shift + mu - sum(arparams[1:]*(y[(i-1):(i-p0-1):(-1)] - mu)) \
    #    - ARparams[1] * sum(arparams*(y[(i-s):(i-s-p0-1):(-1)] - mu)) \
    #    - ARparams[2] * sum(arparams*(y[(i-2*s):(i-2*s-p0-1):(-1)] - mu)) \
    #    + sum(maparams*epsilon[i:(i-q0-1):(-1)]) \
    #    + MAparams[1] * sum(maparams*epsilon[(i-s):(i-s-q0-1):(-1)])
    #    y = np.append(y, yi)
    for i in range(tau, N):
        e = np.random.normal(loc = mu, scale = sd, size = 1)
        epsilon = np.append(epsilon, e)
        yi = (mu + shift) - sum(arparams[1:]*(y[(i-1):(i-p0-1):(-1)] - (mu + shift))) \
        - ARparams[1] * sum(arparams*(y[(i-s):(i-s-p0-1):(-1)] - (mu + shift))) \
        - ARparams[2] * sum(arparams*(y[(i-2*s):(i-2*s-p0-1):(-1)] - (mu + shift))) \
        + sum(maparams*epsilon[i:(i-q0-1):(-1)]) \
        + MAparams[1] * sum(maparams*epsilon[(i-s):(i-s-q0-1):(-1)])
        y = np.append(y, yi)

    Y = y

    y_star = [Y[0:2*s]]
    fit = [Y[0:2*s]]
    y = np.zeros(2*s).tolist()
    for l in range((2*s + 1), (N + 1)):
        y_star = np.append(y_star, Y[l-1])
        pred = mu + sum(phi*(y_star[(l-2):(l-p-2):(-1)] - mu)) \
        + Phi[0] * sum(np.append(np.array([1]), -phi)*(y_star[(l-s-1):(l-s-p-2):(-1)] - mu)) \
        + sum(theta*y[(l-2):(l-q-2):(-1)]) \
        + Theta[0] * sum(np.append(np.array([1]), theta)*y[(l-s-1):(l-s-q-2):(-1)])
        fit = np.append(fit, pred)
        y = np.append(y, y_star[l-1] - pred)

    plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(18, 4.0*3))
    plt.subplot(3,1,1)
    plt.ylabel("Original")
    plt.plot(Y, color = "black")
    plt.subplot(3,1,2)
    plt.ylabel("Zoom in")
    plt.plot(Y[:6*s], color = "black")
    plt.subplot(3,1,3)
    plt.ylabel("Residuals")
    plt.plot(y, color = "black")
    """
