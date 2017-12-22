#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import pandas as pd


models = [10, 18, 24, 26]
# fp = [g1_thresh_std, cd, g2_std, l_std, r, r_std, k_std, d_std]
fv = [r'$\sigma_{s}/\langle\tilde{\Delta}\rangle$', '$t/t_{db}$', '$\sigma_{t}/t_{db}$', '$\sigma_{\lambda}/\lambda$', r'$\langle x\rangle$', r'$\sigma_{x}/\langle x\rangle$', '$\sigma_{K}/K$', r'$\sigma_{\Delta}/\langle\tilde{\Delta}\rangle$']
# This details all the variables in the above models. Which variables are considered will differ slightly per model.
model_vars = [[0, 1, 2, 3, 6], [0, 1, 2, 3, 7], [0, 4, 5, 7], [0, 4, 5, 6]]

models_descr = ['Noisy synthesis rate G2 timer', 'Noisy integrator G2 timer', 'Noisy integrator r sensor']
# Here we populate our dataframe based on the simple outputs from the initial matplotlib files.

celltype = ['Mothers', 'Daughters']
obs_types = ['$V_b$ $V_d$ slope', '$% Popn [W]_d<1.0$', '$% Popn [W]_b<1.0$']
meas_types = ['max', 'min']
# obs_types = ['$V_b$ $V_d$ slope']

for i0 in range(len(models)):
    temp_cols = [obj for obj in obs_types]
    temp_cols.insert(0, 'celltype')
    temp_cols.insert(0, 'Model Parameter $y$')
    temp_cols.insert(0, 'type')
    print temp_cols
    df = pd.DataFrame(columns=temp_cols)  # one data frame per model.
    # tmp = np.mean(np.load('./April17_simulation_dilution_asymmetric_model_{0}.npy'.format(models[i0])), axis=-4)
    tmp = np.mean(np.load('./April17_simulation_dilution_asymmetric_model_{0}_revised.npy'.format(models[i0])), axis=-4)
    for i1 in range(len(model_vars[i0])):  # going through each variable respectively
        for i1a in range(len(celltype)):  # number of celltypes
            temp = tmp[:, ..., i1a, 1, :]  # only look at this for discretized time simulations and for one celltype
            # at a time.
            temp1 = np.squeeze(np.split(temp, [1], axis=i1)[0])
            # splits along this axis so that we can take only the value when this variable is zero
            # print np.split(temp, [1], axis=i1)[0].shape

            # temp3 = np.split((np.amax(temp, axis=i1)-temp1)/temp1, [1, 2], axis=-1)
            temp3 = np.split((np.amax(temp, axis=i1) - temp1), [1, 2], axis=-1)

            # print np.amax(temp, axis=i1)[0, 0, 3, 0, 0], temp.shape
            # print np.amin(temp, axis=i1)[0, 0, 3, 0, 0], temp.shape
            # print temp1[0, 0, 3, 0, 0], temp1.shape
            # print temp[:, 0, 0, 3, 0, 0]
            # print temp3[0][0, 0, 3, 0]
            # max percentage change wrt the minimum value for that variable

            # temp4 = np.split((np.amin(temp, axis=i1)-temp1)/temp1, [1, 2], axis=-1)
            temp4 = np.split((np.amin(temp, axis=i1) - temp1), [1, 2], axis=-1)

            # print temp4[0][0, 0, 3, 0]
            # min percentage change wrt the minimum value for that variable
            temp5 = []
            temp5.append([np.ndarray.flatten(obj) for obj in temp3])  # each entry in this list should have the same length,
            # and should be for a different kind of observable
            temp5.append([np.ndarray.flatten(obj) for obj in temp4])  # each entry in this list should have the same length,
            # and should be for a different kind of observable
            for i2 in range(len(temp5[0])):
                # print temp5[0][i2].shape
                if i2 == 0:
                    L1 = temp5[0][i2].shape[0]
            for i2 in range(len(temp5[1])):
                # print temp5[1][i2].shape
                if i2 == 0:
                    L2 = temp5[1][i2].shape[0]
            if L1 != L2:
                raise ValueError('Your projections are wrong, L1={0}, L2={1}'.format(str(L1), str(L2)))
            for i2 in range(L1):
                df_list = [[], []]
                for i3 in range(2):
                    temp6 = []
                    for i4 in range(3):  # now we go through and append each new item to our dataframe
                        # print len(temp5[i3][i4])
                        temp6.append(temp5[i3][i4][i2])  # now each row of our data frame contains our three
                        # observables
                    df_list[i3].append(temp6)
                    df_list[i3][0].insert(0, celltype[i1a])
                    df_list[i3][0].insert(0, fv[model_vars[i0][i1]])
                    df_list[i3][0].insert(0, meas_types[i3])  # different kind of calculation based on max or min
                    df_temp = pd.DataFrame(data=df_list[i3], columns=temp_cols)
                    df = df.append(df_temp, ignore_index=True)
                # print df
                # exit()
    # path = './April17_simulation_dilution_asymmetric_model_{0}'.format(models[i0])
    path = './April17_simulation_dilution_asymmetric_model_{0}_revised'.format(models[i0])
    df.to_pickle(path)
    print 'I have done model', models[i0]
# saving our dataframe
