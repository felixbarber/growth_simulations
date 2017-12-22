#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# This loads data from March17_dilution_symmetric_1.py

N = 10000
dt, nstep = 0.01, 500
tvec = nstep*dt*(1.0+np.asarray(range(N)))

vals = np.zeros([4, 2, 2, N])  # mean cv mean log cv log, celltype, vb vd, it num


# for i1 in range(2):
#     for i2 in range(10000):
#         temp1 = np.load('../../Documents/data_storage/celltype_{0}_vb_it_{1}'.format(str(i1), str(i2))+'.npy')
#         temp2 = np.load('../../Documents/data_storage/celltype_{0}_vd_it_{1}'.format(str(i1), str(i2))+'.npy')
#         temp = [temp1, temp2]
#         for i3 in range(2):
#             vals[0, i1, i3, i2] = np.mean(temp[i3])
#             vals[1, i1, i3, i2] = scipy.stats.variation(temp[i3])
#             vals[2, i1, i3, i2] = np.mean(np.log(temp[i3]))
#             vals[3, i1, i3, i2] = scipy.stats.variation(np.log(temp[i3]))
#         del temp, temp1, temp2
# np.save('./March17_dilution_symmetric_2_model18_g2_std_015', vals)  # N for this run was 10000

cells = ['M', 'D']
stat = ['Mean', 'CV', 'Mean log', 'CV log']
dtypes = ['Vb', 'Vd']
for i0 in range(4):
    for i1 in range(2):
        fig = plt.figure(figsize=[8, 8])
        for i2 in range(2):
            plt.plot(tvec[::100], vals[i0, i2, i1, ::100], label=cells[i2])
            temp=scipy.stats.pearsonr(tvec,vals[i0, i2, i1, :])
            print 'Celltype {0}, stat {1}, dtype {2}'.format(str(i2), str(i0), str(i1)), temp
        plt.title(stat[i0]+' '+dtypes[i1], size=20)
        plt.xlabel('Time [$t_d$]', size=16)
        plt.legend()
        fig.savefig('./March17_dilution_symmetric_2_stat_{0}_dtype_{1}.eps'.format(str(i0), str(i1)),bbox_inches='tight'
                    , dpi=fig.dpi)
        del fig

# for i0 in range(4):
#     for i1 in range(2):
#         fig = plt.figure(figsize=[8, 8])
#         for i2 in range(2):
#             plt.semilogy(tvec[::100], vals[i0, i2, i1, ::100], label=cells[i2])
#         plt.title(stats[i0]+' '+dtypes[i1], size=20)
#         plt.xlabel('Time [$t_d$]', size=16)
#         plt.legend()
#         fig.savefig('./March17_dilution_symmetric_2_logy_stat_{0}_dtype_{1}.eps'.format(str(i0), str(i1)),
#                     bbox_inches='tight', dpi=fig.dpi)
#         del fig


# This section of code treats the longer simulations run from March17_dilution_symmetric_3.py

# num_rep = 4
# N = 30001
# dt, nstep = 0.01, 500
# tvec = nstep*dt*(1.0+np.asarray(range(N)))
#
# vals = np.load('./March17_dilution_symmetric_3.npy')
#
# for i0 in range(4):
#     for i1 in range(2):
#         fig = plt.figure(figsize=[8, 8])
#         for i2 in range(2):
#             plt.semilogy(tvec[::100], vals[i0, i2, i1, ::100], label=cells[i2])
#         plt.title(stats[i0]+' '+dtypes[i1], size=20)
#         plt.xlabel('Time [$t_d$]', size=16)
#         plt.legend()
#         fig.savefig('./March17_dilution_symmetric_3_logy_stat_{0}_dtype_{1}.eps'.format(str(i0), str(i1)),
#                     bbox_inches='tight', dpi=fig.dpi)
#         del fig
