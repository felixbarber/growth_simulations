#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# this script takes the saved populations from March17_dilution_symmetric_3 and grows them up, generating a data_set
# that tracks the standard deviations over time.

td = 1.0
delta = 10.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('CD', 1.0),
                 ('g2_std', 0.2), ('d_std', 0.1), ('g1_thresh_std', 0.1)])
num_rep = 8
N = 50001
# N=10
save_freq = 1000
# save_freq = 2
num_saves = N/save_freq+1  # gives the number of different saves that this should go through
X = [num_rep, num_saves, 2, 8]
a = np.zeros(X)

tic = time.clock()
for i0 in range(num_rep):  # varying the different growth conditions
    for i1 in range(num_saves):
        temp = np.load('../../Documents/data_storage/March17_dilution_symmetric_3_savedpop_model_{0}_rep_{1}_it_{2}_v1.npy'.format(str(par_vals['modeltype']),
                                                                                        str(i0), str(save_freq*i1)))
        temp0 = g.from_stored_pop(temp, par_vals)
        new_c = g.discr_time_1(par_vals, temp0)
        temp1 = [obj for obj in new_c[0] if obj.exists]
        for i2 in range(2):
            temp2 = np.asarray([obj.vb for obj in temp1 if obj.isdaughter == i2])
            temp3 = np.asarray([obj.wb for obj in temp1 if obj.isdaughter == i2])
            a[i0, i1, i2, 0] = np.mean(temp2)
            a[i0, i1, i2, 1] = np.std(temp2)
            a[i0, i1, i2, 2] = np.mean(temp3)
            a[i0, i1, i2, 3] = np.std(temp3)
            a[i0, i1, i2, 4] = np.mean(np.log(temp2))
            a[i0, i1, i2, 5] = np.std(np.log(temp2))
            a[i0, i1, i2, 6] = np.mean(np.log(temp3))
            a[i0, i1, i2, 7] = np.std(np.log(temp3))
            del temp2, temp3
    print 'I have done {0} sets of repetitions'.format(i0)
    np.save('./April17_paper_scripts_6_model_{0}_v1'.format(par_vals['modeltype']), a)

# v1 prevents negative growth, but comes from saved populations which have incomplete information since they came from
# g.popn_sample_storage
