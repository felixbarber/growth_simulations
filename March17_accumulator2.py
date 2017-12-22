#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
import pandas as pd
import growth_simulation_accumulator_asymmetric as h

# takes input from Feb17_test_models3.py and converts each growth medium result to a violin plot of the
# observed slopes in a pandas setup

filename = 'March17_accumulator1_V0'
data = np.load(filename+'.npy')

r = np.linspace(0.45, 0.75, 8)
r_std = np.linspace(0.0, 0.28, 8)
g1_thresh_std = np.linspace(0.0, 0.28, 8)
models = [3, 4]
num_rep = 5
num_celltype = 2
num_sims = 2
num_meas = 2

print data[0, 7, 7, 0, 0, 0, 1, 0]
par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3)])
par1['r'] = r[0]
par1['r_std'] = r_std[7]
par1['g1_thresh_std'] = g1_thresh_std[7]
print h.slope_vbvd(par1, celltype=0)


if data.shape[0] != 8 or data.shape[-1] != 2 or data.shape[-2] != 2:
    raise ValueError('You are confusing your files')

lab_0 = list(['Accumulator Negative growth', 'Accumulator No Negative growth'])
lab_1 = ['Mothers', 'Daughters']
lab_2 = ['Discrete Gen', 'Discrete time']

celltype = ['M', 'D']
# X = [len(r), len(r_std), len(g1_thresh_std), len(models), num_rep, num_celltype, num_sims, num_meas]
# a = np.zeros(X)
# Here we populate our dataframe based on the simple outputs from the initial matplotlib files.
lab = ['full label', 'Model', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
df = pd.DataFrame(columns=lab)
for i0 in range(len(lab_0)):
    for i1 in range(len(lab_1)):
        for i2 in range(len(lab_2)):
            name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
            temp0 = data[:, :, :, i0, :, i1, i2, 0]
            temp = np.ndarray.flatten(temp0)  # this version restricts sigma_i=0
            for i3 in range(len(temp)):
                df_list = [[name, lab_0[i0], lab_1[i1], lab_2[i2], temp[i3]]]
                temp1 = pd.DataFrame(data=df_list, columns=lab)
                df = df.append(temp1, ignore_index=True)
                del df_list, temp1
            del temp
        print "I have done " + lab_1[i1]
    print "I have done "+lab_0[i0]
# saving our data frame
path = './'+filename
df.to_pickle(path)
del path
del df
