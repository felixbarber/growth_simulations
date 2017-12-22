#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

# This script can be run to generate data which will test the convergence of the basic simulation with initial popn
# seeded by starting population. The initial parameters are

# par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('wd', 1.0), ('wm', 1.0), ('std_v', 0.2), ('std_w', 0.2)])

r = 0.52
x = 0.2
f = r*2**x/(1+r*2**x)
cd = np.log(1+r)/np.log(2)
tau = 0.0
delta = 10.0

models = [17, 17, 5, 15]
model_descr = ['Noisy integrator', 'Noiseless integrator', 'Noisy synth', 'Noisy synth const frac']

# These parameter settings are defined anew for each different model.
par1_vals = [[['modeltype', 17], ['d_std', 0.2], ['num_gen', 9]], [['modeltype', 17], ['d_std', 0.0], ['num_gen', 9]],
             [['modeltype', 5], ['k_std', 0.2], ['num_gen', 10]], [['modeltype', 15], ['w_frac', f], ['k_std', 0.2],
                                                                   ['num_gen', 10]]]

g1_std = np.linspace(0.0, 0.2, 2)
g2_std = np.linspace(0.0, 0.2, 6)
num_obs = 6
num_types = 3

X0, X1, X2, X3 = len(g1_std), len(g2_std), num_obs, num_types

par_set = dict([('g1_std', 0.0), ('nstep', 400), ('dt', 0.01), ('CD', cd), ('K', delta / cd), ('td', 1.0),
                 ('g1_delay', 0.0), ('l_std', 0.2), ('delta', delta), ('mothervals', True)])


for i0 in range(len(models)):
    a = np.empty([X0, X1, X2, X3])
    # Reset par1 for each new variable setting
    par1 = par_set
    for temp in range(len(par1_vals[i0])):
        par1[par1_vals[i0][temp][0]] = par1_vals[i0][temp][1]
    for i1 in range(X0):
        par1['g1_thresh_std'] = g1_std[i1]
        for i2 in range(X1):
            par1['g2_std'] = g2_std[i2]
            obs = g.single_par_meas6(par1)
            a[i1, i2, :, :] = obs
    np.save('./Feb_files/Feb17_test_models' + str(i0)+'V5', a)
    print "Done model: " + str(i0)

celltypes = ['m', 'd']
for i0 in range(len(models)):
    # Reset par1 for each new variable setting
    par1 = par_set

    for temp in range(len(par1_vals[i0])):
        par1[par1_vals[i0][temp][0]] = par1_vals[i0][temp][1]
    a = np.load('./Feb_files/Feb17_test_models'+str(i0)+'V5'+'.npy')
    figs = g.test_function_syst(a, par1, g1_std, g2_std * par1['td'], vec=range(len(g1_std)))
    u = 1
    for i1 in range(len(figs)):  # note that test_function_syst gives back daughter plot first, then mother plot.
        figs[i1].savefig('./Feb_files/Feb17_test_models' + str(i0) + celltypes[u] + 'V5' + '.eps', bbox_inches='tight', dpi=figs[i1].dpi)
        u += -1
    del figs
