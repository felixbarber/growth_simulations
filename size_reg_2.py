#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_bacteria_initiator as g
import time
from scipy import stats
import scipy


par1 = dict([('td', 1.0), ('modeltype', 2), ('g1_thresh_std', 0.02), ('num_gen1', 9), ('num_gen', 9), ('delta', 2.0), ('CD', 0.7),
             ('g2_std', 0.01), ('dt', 0.01), ('nstep', 500), ('num_s1', 500)])
path='./size_reg_2_d.npy'
print par1, path
g1_std = np.linspace(0.0, 0.3, 31)
num_rep = 50
A = np.zeros([len(g1_std), num_rep, 2])

for i0 in range(len(g1_std)):
    par1['g1_thresh_std'] = g1_std[i0]
    for i1 in range(num_rep):
        c = []
        temp1 = g.discr_gen(par1)
        c.append(temp1)
        del temp1
        # This will initialize the subsequent simulations for this model
        temp = g.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
        # initial pop seeded from c
        temp1, obs1 = g.discr_time_1(par1, temp)
        c.append(temp1)
        del temp1, temp

        temp = np.asarray([obj.vb for obj in c[1] if obj.exists])
        temp1 = np.asarray([obj.vd for obj in c[1] if obj.exists])
        temp2 = scipy.stats.linregress(temp, temp1)  # slope based on leaf cells
        A[i0, i1, 0] = temp2[0]
        x = obs1[1][100:]
        temp3 = scipy.stats.linregress(x, np.log(obs1[0][100:]))  # population growth rate
        A[i0, i1, 1] = temp3[0]
    print par1['g1_thresh_std'], i0
np.save(path, A)


