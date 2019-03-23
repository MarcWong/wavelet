import numpy as np
from wavelet.plot import wavelet # 新方法

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

for i in range(0, len(name)):
    X = np.load("../silicon/" + name[i])
    X_output, y_baseline = wavelet(X, LEVEL)

    print("保存文件: ../silicon_data/" + name[i])
    np.save("../silicon_data/" + name[i], X_output )
    # print("保存文件: ../silicon_data_y/" + name[i])
    # np.save("../silicon_data_y/" + name[i], y_baseline )