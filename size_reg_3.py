#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_bacteria_initiator as g
import time
from scipy import stats
import scipy

# par1 copied directly from size_reg_2.py
par1 = dict([('td', 1.0), ('modeltype', 2), ('g1_thresh_std', 0.02), ('num_gen1', 9), ('num_gen', 9), ('delta', 2.0),
             ('CD', 0.7),
             ('g2_std', 0.2), ('dt', 0.01), ('nstep', 500), ('num_s1', 500)])
path1, path2, path3, path4 = './size_reg_2.npy', './size_reg_2_b.npy', './size_reg_2_c.npy', './size_reg_2_d.npy'
data1 = np.load(path1)
data2 = np.load(path2)
data3 = np.load(path3)
data4 = np.load(path4)
temp = np.concatenate((data1, data2), axis=1)
data = np.concatenate((temp, data3), axis=1)
del data1, data2, data3

g1_std = np.linspace(0.0, 0.3, 31)
av = np.mean(data, axis=1)  # sigma_t=0.2
av1 = np.mean(data4, axis=1)  # sigma_t=0.01


fig = plt.figure(figsize=[6, 6])
plt.plot(g1_std, (av[:, 1]-np.log(2)/par1['td'])/av[:, 1], label=r'$\sigma_t/t_{db}=0.2$')
plt.plot(g1_std, (av1[:, 1]-np.log(2)/par1['td'])/av1[:, 1], label=r'$\sigma_t/t_{db}=0.01$')
plt.legend()
plt.xlabel(r'$\sigma_i/\langle A_c\rangle$')
plt.ylabel(r'$\frac{\Lambda_p-\lambda}{\Lambda_p}$')
fig.savefig('./size_reg_3_gr.eps', bbox_inches='tight', dpi=fig.dpi)
del fig

fig = plt.figure(figsize=[6, 6])
plt.plot(g1_std, av[:, 0], label='$\sigma_t/t_{db}=0.2$')
plt.plot(g1_std, av1[:, 0], label='$\sigma_t/t_{db}=0.01$')
plt.legend()
plt.xlabel(r'$\sigma_i/\langle A_c\rangle$')
plt.ylabel('$V_b$ vs. $V_d$ slope')
fig.savefig('./size_reg_3_vbvdslope.eps', bbox_inches='tight', dpi=fig.dpi)
del fig
