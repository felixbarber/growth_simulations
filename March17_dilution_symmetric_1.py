#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

td = 1.0
delta = 10.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('K', delta/td),
             ('CD', 1.0), ('r', 1.0), ('r_std', 0.0), ('g2_std', 0.2), ('d_std', 0.1), ('k_std', None),
                 ('g1_thresh_std', 0.1)])
cd = [0.5, 1.0]
vb = [[[], []], [[], []]]
vd = [[[], []], [[], []]]

# for i0 in range(len(cd)):
i0 = 1
par1 = par_vals
par1['CD'] = cd[i0]
c = g.discr_gen(par_vals)
temp = [obj for obj in c if obj.exists]
# del c
for i1 in range(100000):
    val = g.starting_popn_seeded(temp, par1)
    c = g.discr_time_1(par1, val)
    temp = [obj for obj in c[0] if obj.exists]
    for i2 in range(2):
        temp1 = np.asarray([obj.vb for obj in temp if obj.isdaughter == i2 and obj.exists])
        temp2 = np.asarray([obj.vd for obj in temp if obj.isdaughter == i2 and obj.exists])
        np.save('../../Documents/data_storage/celltype_{0}'.format(i2)+'_vb_it_{0}'.format(i1), temp1)  # saving data
        np.save('../../Documents/data_storage/celltype_{0}'.format(i2) + '_vd_it_{0}'.format(i1), temp2)  # saving data
        vb[i0][i2].append(temp1)
        vd[i0][i2].append(temp2)


# for i0 in range(len(cd)):
# for i1 in range(2):
#     fig = plt.figure(figsize=[8, 8])
#     for i2 in range(0, 10000, 1000):
#         sns.distplot(vb[i0][i1][i2], label='Iteration {0}'.format(i2), hist=False, kde=True)
#     plt.legend()
#     fig.savefig('./March17_dilution_symmetric_1_cd_{0}'.format(i0)+'_celltype_{0}'.format(i1)+'.eps',
#                 bbox_inches='tight', dpi=fig.dpi)
