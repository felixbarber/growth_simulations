#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
import pandas as pd

# takes input from Feb17_test_models3.py and converts each growth medium result to a violin plot of the
# observed slopes in a pandas setup

# load the output of Feb17_test_models3.py that was done with the accurate estimates of biological parameters
filename = 'Feb17_test_models3_model16_V2'
data = np.load(filename+'.npy')

filename_1 = 'Feb17_test_models3_model16'  # this data is not useful. Had the incorrect values estimated for f.
data_1 = np.load(filename_1+'.npy')
# data_av = np.mean(data, axis=-3)

num_med = 5
if data.shape[0] != 5 or data.shape[-1] != 3 or data.shape[-2] != 2:
    raise ValueError('You are confusing your files')
if data_1.shape[0] != 5 or data_1.shape[-1] != 3 or data_1 .shape[-2] != 2:
    raise ValueError('You are confusing your files')
# X = [num_med, len_var, len_unfx, len_var, len_var, len_unfx, len_var, num_rep, num_celltype, num_sims]

lab_0 = list(['Glucose', 'Galactose', 'Glycerol', 'Low Glucose', 'Raffinose'])
lab_1 = ['Mothers', 'Daughters']
lab_2 = ['Discrete Gen', 'Discrete time', 'Theory']

doubling_time_vec = np.array([89, 130, 220, 100, 120, 100, 135, 190, 120])
celltype = ['M', 'D']

# Here we populate our dataframe based on the simple outputs from the initial matplotlib files.
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']

# full dataset version

# df = pd.DataFrame(columns=lab)
# for i0 in range(len(lab_0)):
#     for i1 in range(len(lab_1)):
#         for i2 in range(len(lab_2)):
#             name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
#             temp = np.ndarray.flatten(data[i0, ..., i1, i2])
#             for i3 in range(len(temp)):
#                 df_list = [[name, lab_0[i0], lab_1[i1], lab_2[i2], temp[i3]]]
#                 temp1 = pd.DataFrame(data=df_list, columns=lab)
#                 df = df.append(temp1, ignore_index=True)
#                 del df_list, temp1
#             del temp
#         print "I have done " + lab_1[i1]
#     print "I have done "+lab_0[i0]
# # saving our data frame
# path = './'+filename
# df.to_pickle(path)
#
# del df

# This version is for the zero G1_thresh_std version
#
# df = pd.DataFrame(columns=lab)
# for i0 in range(len(lab_0)):
#     for i1 in range(len(lab_1)):
#         for i2 in range(len(lab_2)):
#             name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
#             temp0 = np.mean(data[i0, :, 0, :, :, 3, :, :, i1, i2], axis=-1)
#             temp = np.ndarray.flatten(temp0)  # this version restricts sigma_i=0
#             for i3 in range(len(temp)):
#                 df_list = [[name, lab_0[i0], lab_1[i1], lab_2[i2], temp[i3]]]
#                 temp1 = pd.DataFrame(data=df_list, columns=lab)
#                 df = df.append(temp1, ignore_index=True)
#                 del df_list, temp1
#             del temp
#         print "I have done " + lab_1[i1]
#     print "I have done "+lab_0[i0]
# # saving our data frame
# filename_2 = 'Feb17_test_models3_model16_V2_si_0'
# path = './'+filename_2
# df.to_pickle(path)
# del path, filename_2
#
# del df

# This version is for version 3, with adjusted ranges for s_i and k_std
filename = 'Feb17_test_models3_model16_V3'
data = np.load(filename+'.npy')
df = pd.DataFrame(columns=lab)
for i0 in range(len(lab_0)):
    for i1 in range(len(lab_1)):
        for i2 in range(len(lab_2)):
            name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
            temp0 = np.mean(data[i0, :, :, :, :, :3, :, :, i1, i2], axis=-1)
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
del path, filename

del df

# # This version is for version 3, with adjusted ranges for s_i and k_std
# filename = 'Feb17_test_models3_model16_V3'
# data = np.load(filename+'.npy')
# df = pd.DataFrame(columns=lab)
# for i0 in range(len(lab_0)):
#     for i1 in range(len(lab_1)):
#         for i2 in range(len(lab_2)):
#             name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
#             temp0 = np.mean(data[i0, :, 0, :, :, 2:, :, :, i1, i2], axis=-1)
#             temp = np.ndarray.flatten(temp0)  # this version restricts sigma_i=0
#             for i3 in range(len(temp)):
#                 df_list = [[name, lab_0[i0], lab_1[i1], lab_2[i2], temp[i3]]]
#                 temp1 = pd.DataFrame(data=df_list, columns=lab)
#                 df = df.append(temp1, ignore_index=True)
#                 del df_list, temp1
#             del temp
#         print "I have done " + lab_1[i1]
#     print "I have done "+lab_0[i0]
# # saving our data frame
# path = './'+filename+'_constr'
# df.to_pickle(path)
# del path, filename
#
# del df

# old (and incorrect) data set.

# df = pd.DataFrame(columns=lab)
# for i0 in range(len(lab_0)):
#     for i1 in range(len(lab_1)):
#         for i2 in range(len(lab_2)):
#             name = lab_0[i0]+', '+lab_1[i1]+', '+lab_2[i2]
#             temp = np.ndarray.flatten(data_1[i0, ..., i1, i2])
#             for i3 in range(len(temp)):
#                 df_list = [[name, lab_0[i0], lab_1[i1], lab_2[i2], temp[i3]]]
#                 temp1 = pd.DataFrame(data=df_list, columns=lab)
#                 df = df.append(temp1, ignore_index=True)
#                 del df_list, temp1
#             del temp
#         print "I have done " + lab_1[i1]
#     print "I have done "+lab_0[i0]
# # saving our data frame
# path_1 = './'+filename_1+'_old'
# df.to_pickle(path_1)
