import numpy as np

def split(original_path, gt_path, filename):
    X = np.load(original_path + filename + ".npy")
    y = np.load(gt_path + filename + ".npy")

    np.save(original_path + filename + "_train.npy", X[1::2])
    np.save(original_path + filename + "_test.npy", X[::2])

    np.save(gt_path + filename + "_train.npy", y[1::2])
    np.save(gt_path + filename + "_test.npy", y[::2])
    return