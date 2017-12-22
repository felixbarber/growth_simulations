#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib
# matplotlib.use('GTK')
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# Now we do plotting for March17_dilution_symmetric_3.py
data = np.load('March17_dilution_symmetric_3.npy')
data_1 = np.load('March17_dilution_symmetric_3_asymm.npy')

td = 1.0
delta = 10.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('CD', 1.0),
                 ('g2_std', 0.2), ('d_std', 0.1), ('g1_thresh_std', 0.1)])
celltype = ['Mothers', 'Daughters']
num_rep = 4
N = 50001
# N=10
save_freq = 1000
X = [num_rep, N, 2, 8]
# temp2 = np.asarray([obj.vb for obj in temp if obj.isdaughter == i2 and obj.exists])
# temp3 = np.asarray([obj.wb for obj in temp if obj.isdaughter == i2 and obj.exists])
# a[i0, i1, i2, 0] = np.mean(temp2)
# a[i0, i1, i2, 1] = np.std(temp2)
# a[i0, i1, i2, 2] = np.mean(temp3)
# a[i0, i1, i2, 3] = np.std(temp3)
# a[i0, i1, i2, 4] = np.mean(np.log(temp2))
# a[i0, i1, i2, 5] = np.std(np.log(temp2))
# a[i0, i1, i2, 6] = np.mean(np.log(temp3))
# a[i0, i1, i2, 7] = np.std(np.log(temp3))

tvec = np.linspace(1, N, N)*par_vals['nstep']*par_vals['dt']
for i0 in range(2):
    fig = plt.figure(figsize=[10, 6])
    for i1 in range(data.shape[0]):
        plt.plot(tvec[::1000], data[i1, ::1000, i0, 7], alpha=0.2)
        plt.plot(tvec[::1000], data_1[i1, ::1000, i0, 7], alpha=0.2)
    plt.plot(tvec[::1000], np.mean(data[:, ::1000, i0, 7], axis=0), label=celltype[i0] + ' t=t_db mean')
    plt.plot(tvec[::1000], np.mean(data_1[:,::1000,i0,7], axis=0),label=celltype[i0] + ' t=0.6t_db, mean')
    if i0==1:
        for i2 in range(data_1.shape[0]):
            print data_1[i2, :25, i0, 7]
            print data[i2, :25, i0, 7]
    plt.xlabel(r'Time [$t_{db}$]', size=20)
    plt.ylabel(r'$\sigma (\log(V_b))$', size=20)
    plt.xlim(xmin=0.0, xmax=2.5*10**5)
    plt.title(r'$\sigma (\log(V_b))$ vs. time for a symmetrically dividing population', size=20)
    plt.legend(loc=1)
    fig.savefig('./March17_dilution_symmetric_plotter_1_model_{0}_celltype_{1}.png'.format(str(par_vals['modeltype']),str(i0)), dpi=fig.dpi)
    del fig
    print i0
# fig = plt.figure(figsize=[10, 6])
# for i0 in range(2):
#     for i1 in range(data.shape[0]):
#         plt.loglog(tvec[::10], data[i1, ::10, i0, 7], label=celltype[i0]+' rep. {0}'.format(i1), alpha=0.2)
#         plt.loglog(tvec[::10], data_1[i1, ::10, i0, 7], label=celltype[i0] + r' $t=0.6t_{db}$, rep. {0}'.format(i1), alpha=0.2)
#     plt.loglog(tvec[::10],np.mean(data[:,::10,i0,7],axis=0),label=celltype[i0]+' mean')
#     plt.loglog(tvec[::10], np.mean(data_1[:, ::10, i0, 7], axis=0), label=celltype[i0] + r' $t=0.6t_{db}$, mean')
# plt.xlabel(r'Time [$t_{db}$]', size=20)
# plt.ylabel(r'$\sigma (\log(V_b))$', size=20)
# plt.title(r'$\sigma (\log(V_b))$ vs. time for a symmetrically dividing population', size=20)
# plt.legend()
# fig.savefig('./March17_dilution_symmetric_plotter_1_model_{0}_loglog.png'.format(par_vals['modeltype']), dpi=fig.dpi)
