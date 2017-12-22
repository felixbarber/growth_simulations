#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

data = np.load('save_data_2.npy')

delta = 10.0

par1 = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
                ('num_gen', 9), ('modeltype', 24), ('delta', delta)])

# X = [len(r), len(r_std), len(g1_thresh_std), len(d_std), num_rep, num_celltype, num_sims, num_meas]

model_descr = ['Noisy integrator no neg growth']
r = np.linspace(0.45, 0.75, 8)
r_std = np.linspace(0.0, 0.28, 8)
d_std = np.linspace(0.0, 0.2, 2)
g1_thresh_std = np.linspace(0.0, 0.2, 2)
num_rep = 3
num_celltype = 2
num_sims = 2
num_meas = 3
filename = 'March17_test_models4_model'+str(par1['modeltype'])+'_V0'
data = np.load(filename+'.npy')

labels = ['$\sigma_{r}/r$', 'r', ' Noisy integrator with $\sigma_{i}/<\Delta>=$', '$\sigma_{i}/<\Delta>=$']
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent [Whi5]$<1$ at division', 'Percent [Whi5]$<1$ at birth']
celltype = ['Mothers', 'Daughters']
sim_type = ['Discr gen ', 'Discr time ']
for i0 in range(len(g1_thresh_std)):
    for i1 in range(len(d_std)):
        for i2 in range(num_meas):
            for i3 in range(num_sims):
                fig = plt.figure(figsize=[16, 8])
                for i4 in range(num_celltype):
                    ax = plt.subplot(1, 2, 1+i4)
                    obs = np.mean(data[:, :, i0, i1, :, i4, i3, i2], axis=-1)
                    ax = g.heat_map(obs, r, r_std, ax, xlabel=labels[0], ylabel=labels[1], title=celltype[i4], fmt='.3g')
                temp = sim_type[i3]+meas_type[i2] + labels[2] + str(g1_thresh_std[i0]) + \
                            ' $\sigma_{\Delta}/\Delta=$' + str(d_std[i1])
                plt.suptitle(temp, size=20)
                fig.savefig('./March17_test_models5_model24_meas_'+str(i2)+'_si_' + str(i0) + '_sd_' + str(i1) + '_sim_'
                            +str(i3) + '.eps', bbox_inches='tight', dpi=fig.dpi)
                del fig, obs, ax
                if i3 == 1:
                    fig = plt.figure(figsize=[8, 8])

                    # plots for the paper. Daughters only. Individual plots so can be mixed and matched.
                    # Discretized time only.
                    ax = plt.subplot(1, 1, 1)
                    i4 = 1  # daughters only
                    obs = np.mean(data[:, :, i0, i1, :, i4, i3, i2], axis=-1)
                    temp = labels[3] + str(g1_thresh_std[i0]) + \
                           ', $\sigma_{\Delta}/<\Delta>=$' + str(d_std[i1])
                    ax = g.heat_map(obs, r, r_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
                                    fmt='.3g')
                    fig.savefig(
                        './March17_test_models5_model24_D_meas_' + str(i2) + '_si_' + str(i0) + '_sd_' + str(i1) + '_sim_'
                        + str(i3) + '.eps', bbox_inches='tight', dpi=fig.dpi)
        # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
        # will be modified.

