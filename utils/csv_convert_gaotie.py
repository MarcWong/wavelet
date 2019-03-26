import numpy as np

filename = [
    '2327_20170131-03-zs',
    '2327_20170201-03-zs',
    '2327_20170204-03-zs',
    '2336_20170210-10-zs',
    '2437_20161108-07-zs',
    '2544_20160907-13-zs',
    '2587_20161109-02-zs',
    '2587_20161220-12-zs',
    '2618_20160723-03-zs',
    '2618_20170102-07-zs',
    '2658_20161002-10-zs',
    '2661_20161211-07-zs',
    '2678_20161209-06-zs',
    '2778_20170127-07-zs'
]

filename2 = [
    '2341_20161221-04-zs',
    '2651_20160624-01-zs',
]

for i in range(0, len(filename)):
    raw_data_name = '../../data/' + str(filename[i]) + '.csv'
    raw_data = np.loadtxt(raw_data_name, delimiter=",", skiprows=1)


    for x in range(6,42):
        # 取某一个车厢的传感器
        data = raw_data[:,x]

        output_name = '../../data_raw/' + str(filename[i]) + '/' + str(x - 5) +'.npy'
        np.save(output_name, data)


for i in range(0, len(filename2)):
    raw_data_name = '../../data/' + str(filename2[i]) + '.csv'
    raw_data = np.loadtxt(raw_data_name, delimiter=",", skiprows=1)


    for x in range(6,13):
        # 取某一个车厢的传感器
        data = raw_data[:,x]

        output_name = '../../data_raw/' + str(filename2[i]) + '/' + str(x - 5) +'.npy'
        np.save(output_name, data)