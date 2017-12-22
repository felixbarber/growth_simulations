#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np

labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

g1_std = np.linspace(0.0, 0.25, 6)
g2_std = np.linspace(0.0, 0.25, 6)
cd = np.linspace(0.5, 1.0, 8)
l_std = np.linspace(0.0, 0.2, 2)
k_std = np.linspace(0.0, 0.2, 2)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.1), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 10)
             , ('k_std', 0.0), ('l_std', 0.0)])

# GENERATING MODEL PREDICTIONS
n = 1
g1_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))
g2_std_m = np.linspace(0.0, 0.25, 1+n*(len(g2_std)-1))
cd_m = np.linspace(0.5, 1.0, 1+n*(len(cd)-1))
l_std_m = np.linspace(0.0, 0.2, 1+n*(len(l_std)-1))
k_std_m = np.linspace(0.0, 0.2, 1+n*(len(k_std)-1))

slopes = np.zeros([len(g1_std_m), len(g2_std_m), len(cd_m), len(l_std_m), len(k_std_m), 2])
for h in range(len(g1_std_m)):
    par1['g1_thresh_std'] = g1_std_m[h]
    for i in range(len(cd_m)):
        par1['cd_m'] = cd_m[i]
        for j in range(len(l_std_m)):
            par1['l_std'] = l_std_m[j]
            for k in range(len(k_std_m)):
                par1['k_std'] = k_std_m[k]
                slopes[h, :, i, j, k, 0] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], g2_std_m)
                slopes[h, :, i, j, k, 1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], g2_std_m)
np.save('model'+str(par1['modeltype'])+'_theory_gen_numsamp'+str(n), slopes)
