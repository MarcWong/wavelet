import numpy as np

def calibrate(input_path, output_path, filename):
    X = np.load(input_path + filename + '.npy')

    y_gt = np.zeros(X.shape[0], dtype=int);
    m = 0
    # for i in range(0,1250):
    #     y_gt[i] = 1
    #     m += 1
    for i in range(8346,14577):
        y_gt[i] = 1
        m += 1
    # for i in range(11512,11712):
    #     y_gt[i] = 1
    #     m += 1
    for i in range(13069,19355):
        y_gt[i] = 1
        m += 1

    print("保存标定的真实值:" + output_path + filename + '.npy', "测试集异常点数量: ", int(m / 2))
    np.save(output_path + filename + '.npy', y_gt)
    return
