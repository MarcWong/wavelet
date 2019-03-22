import numpy as np
name = '223-121-非规律'
raw_data_name = '../../silicon/' +name +'.csv'
raw_data = np.float64(np.loadtxt(raw_data_name, delimiter=",", usecols=[2], skiprows=1))
print(raw_data)

output_name = '../../silicon_data/' + name +'.npy'
np.save(output_name, raw_data )