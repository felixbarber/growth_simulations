#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

w_frac = np.linspace(0.2, 0.5, 13)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.15), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 100.0/0.67), ('td', 1.0), ('modeltype', 16), ('l_std', 0.2),
             ('g1_delay', 0.0), ('k_std', 0.0), ('w_frac', 0.0)])
models = [15, 16]
X0, X1 = len(w_frac), len(models)
slopes = np.zeros((X0, X1, 3))
for i0 in xrange(X0):
    par1['w_frac'] = w_frac[i0]
    for i1 in range(X1):
        par1['modeltype'] = models[i1]
        c = g.discr_gen(par1)  # note this also returns slopes for full pop
        for i2 in range(3):  # mothers, daughters and population
            if i2 < 2:
                x1 = [obj.vb for obj in c[1000:] if obj.isdaughter == i2 and obj.wd / obj.vd < 1.0]
                x = [obj.vb for obj in c[1000:] if obj.isdaughter == i2]
                y = [obj.vd for obj in c[1000:] if obj.isdaughter == i2]
            else:
                x1 = [obj.vb for obj in c[1000:] if obj.wd / obj.vd < 1.0]
                x = [obj.vb for obj in c[1000:]]
                y = [obj.vd for obj in c[1000:]]
            slopes[i0, i1, i2] = len(x1) * 100.0 / len(x)
np.save('./lab_meeting_figures/model_'+str(models[0])+str(models[1])+'_numneg', slopes)
