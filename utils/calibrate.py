import numpy as np

def calibrate(input_path, output_path, filename):
    X = np.load(input_path + filename + '.npy')
    print(X.shape)
    y_gt = np.zeros(X.shape[0]);
    for i in range(3250,8400):
        y_gt[i] = 1;

    print("保存标定的真实值:" + output_path + filename + '.npy', "异常值范围: 3250-8400")
    np.save(output_path + filename + '.npy', y_gt)
    return