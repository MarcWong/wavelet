import numpy as np
from wavelet.plot import wavelet # 新方法

def process_wavelet(input_path, output_path, filename, LEVEL):
    X = np.load(input_path + filename + '.npy')
    X_output, y_baseline = wavelet(X, LEVEL)
    print("保存文件:" + output_path + filename + '.npy')
    np.save(output_path + filename, X_output )

    Y_baseline_path = output_path.replace('data', 'data_gt')
    Y_baseline_path = Y_baseline_path.replace('.npy', '')

    print("保存ewma结果:" + Y_baseline_path + filename + '_train_ewma.npy,与' + Y_baseline_path + filename + '_test_ewma.npy')

    np.save(Y_baseline_path + filename + "_train_ewma.npy", y_baseline[1::2])
    np.save(Y_baseline_path + filename + "_test_ewma.npy", y_baseline[::2])