import numpy as np

# name = ['223-94.npy',
#     '223-95.npy',
#     '223-96.npy',
#     '223-97.npy',
#     '223-100-问题.npy',
#     '223-101.npy',
#     '223-105-问题.npy',
#     '223-106.npy',
#     '223-107-小问题.npy',
#     '223-108.npy',
#     '223-109-断.npy',
#     '223-111-非规律.npy',
#     '223-112-小问题.npy',
#     '223-114-非规律.npy',
#     '223-115-非规律.npy',
#     '223-117.npy',
#     '223-118.npy',
#     '223-119-问题.npy',
#     '223-120.npy',
#     '223-121-非规律.npy',
# ]

name = '223-121-非规律'
raw_data_name = '../../silicon_raw/' +name +'.csv'
raw_data = np.float64(np.loadtxt(raw_data_name, delimiter=",", usecols=[2], skiprows=1))
print(raw_data)

output_name = '../../silicon/' + name +'.npy'
np.save(output_name, raw_data )