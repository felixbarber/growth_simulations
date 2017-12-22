#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy


r = 0.55
x = 0.2

cd = np.log(1+r)/np.log(2)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.0), ('dt', 0.01)
            , ('CD', cd), ('num_gen', 9), ('K', 10.0/cd), ('td', 1.0), ('modeltype', 20),
             ('g1_delay', x), ('l_std', 0.2), ('d_std', 0.2), ('delta', 10.0)])

print par1

obs = g.single_par_meas2(par1)
i0=1
s_i = par1['g1_thresh_std']
s_cd = par1['g2_std']*par1['td']
label=['Slope', '<Vb>', '<VbVb>', '<Vd>', '<VdVb>', '<Wb>', '<WbWb>', 'Fraction low conc']
obs1 = [g.slope_vbvd_func(par1, s_i, s_cd), g.vb_func(par1, s_i, s_cd), g.vbvb_func(par1, s_i, s_cd), \
        g.vd_func(par1, s_i, s_cd), g.vdvb_func(par1, s_i, s_cd), g.wb(par1, cell_no=i0), g.wbwb(par1, cell_no=i0)]
print "Daughters"
for j in range(obs.shape[0]-1):
    print label[j], np.round(obs[j, i0], 3), np.round(obs1[j], 3)
print label[7], np.round(obs[7, i0], 3)
i0 = 0
print "Mothers"
sel = [1, 5, 6]
obs2 = [g.vb_m(par1, s_i, s_cd), g.wb(par1, cell_no=i0), g.wbwb(par1, cell_no=i0)]
for j in range(len(sel)):
    print label[sel[j]], np.round(obs[sel[j], i0], 3), np.round(obs2[j], 3)
print label[7], np.round(obs[7, i0], 3)
