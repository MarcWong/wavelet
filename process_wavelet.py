import numpy as np
from wavelet.plot import wavelet # 新方法

def process_wavelet(inputName, outputName, LEVEL):
    X = np.load(inputName)
    X_output, y_baseline = wavelet(X, LEVEL)

    print("保存文件:" + outputName)
    np.save(outputName, X_output )