#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
import scipy
from scipy import stats

# This script will consider the effect of initialization on the simulations.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 50 parallel simulations, both discretized time and discretized gen.
# We then look at the mean and standard deviation of the resultant observed slopes and test their deviation from the
# theoretically predicted slopes.

r = 0.52
x = 0.2
f = r*2**x/(1+r*2**x)
cd = np.log(1+r)/np.log(2)
delta = 10.0

par_set = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('dt', 0.01), ('CD', cd), ('K', delta/cd),
                ('td', 1.0), ('g1_delay', 0.0), ('l_std', 0.2), ('d_std', 0.0), ('delta', delta)])

# These parameter settings are defined anew for each different model.
par1_vals = [[['modeltype', 17], ['d_std', 0.2], ['num_gen', 10]], [['modeltype', 17], ['d_std', 0.0], ['num_gen', 10]],
             [['modeltype', 5], ['k_std', 0.2], ['num_gen', 10]], [['modeltype', 15], ['w_frac', f], ['k_std', 0.2],
                                                                   ['num_gen', 10]]]
models = [17, 17, 5, 15]
model_descr = ['Noisy integrator', 'Noiseless integrator', 'Noisy synth', 'Noisy synth const frac']

num_s1 = [100, 200, 300, 400, 500, 600, 700]
num_gen1 = [3, 4, 5, 6, 7]
num_rep = 100
num_celltype = 2
celltype = ['m', 'd']

vals = ['modeltype',  'num_s1', 'num_gen1']
pars = [models, num_s1, num_gen1]

X0, X1, X2, X3, X4 = len(models), len(num_s1), len(num_gen1), num_rep, num_celltype
# a = np.zeros([X0, X1, X2, X3, X4, 2])
temp = np.load('feb17_test_models.npy')
val = list([])
val.append(np.mean(temp, axis=3))
val.append(np.std(temp, axis=3))
del temp

simtype = ['Discr Time', 'Discr Gen']

y_label = None
x_label = None
y_label1 = 'Seed population size'
x_label1 = 'Number of Generations / Growth time (in units of td)'

for i0 in range(X0):
    par1 = par_set
    for temp in range(len(par1_vals[i0])):
        par1[par1_vals[i0][temp][0]] = par1_vals[i0][temp][1]
    temp = list([])
    temp.append(g.slope_vbvd_m(par1, par1['g1_thresh_std'], par1['g2_std']*par1['td']))
    temp.append(g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['g2_std']*par1['td']))
    for i1 in range(X4):
        fig = plt.figure(figsize=[20, 6])
        ind = 1
        for i2 in range(2):
            ax = plt.subplot(1, 4, ind)
            title_str = simtype[i2]+' mean slope diff'
            temp1 = val[0][i0, :, :, i1, i2]
            ax = g.heat_map(temp1-temp[i1], num_s1, num_gen1, ax, xlabel=x_label, ylabel=y_label, title=title_str)
            # ax = g.heat_map(temp1, num_s1, num_gen1, ax, xlabel=x_label, ylabel=y_label, title=title_str)
            ind += 1
            temp1 = val[1][i0, :, :, i1, i2]
            ax = plt.subplot(1, 4, ind)
            title_str = simtype[i2] + ' slope $\sigma$'
            ax = g.heat_map(temp1, num_s1, num_gen1, ax, xlabel=x_label, ylabel=y_label, title=title_str)
            ind += 1
        del ind
        fig.suptitle('Model '+str(models[i0])+celltype[i1]+' Simulations vs. theory', fontsize=20, fontweight='bold')
        fig.savefig('Feb17_test_models2_model_'+str(models[i0])+celltype[i1]+str(i0)+'.eps', bbox_inches='tight', dpi=fig.dpi)
