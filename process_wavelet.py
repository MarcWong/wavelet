import numpy as np
from wavelet.plot import wavelet # 新方法

def process_wavelet(input_path, output_path, filename, LEVEL):
    X = np.load(input_path + filename + '.npy')
    X_output, y_baseline = wavelet(X, LEVEL)

    print("保存文件:" + output_path + filename + '.npy')
    np.save(output_path + filename, X_output )