#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

delta = 10.0


par1 = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
             ('num_gen', 9), ('modeltype', 4), ('delta', delta)])

# X = [len(r), len(r_std), len(g1_thresh_std), len(d_std), num_rep, num_celltype, num_sims, num_meas]

model_descr = ['Noisy integrator no neg growth']
# r = np.linspace(0.45, 1.0, 12)
r = np.linspace(0.5, 0.7, 2)
# r_std = np.linspace(0.0, 0.28, 8)
r_std = np.linspace(0.0, 0.3, 31)
# d_std = np.linspace(0.0, 0.28, 8)
d_std = np.linspace(0.0, 0.3, 31)
g1_thresh_std = np.linspace(0.0, 0.3, 2)
num_rep = 20
num_celltype = 2
num_sims = 2
num_meas = 3
filename = 'inh_yeast_asymm_complete'
data1 = np.load(filename + '.npy')
data2 = np.load(filename + '_1.npy')
data=np.concatenate((data1, data2), axis=4)
print data.shape

labels = [r'$\sigma_{\Delta}/\langle\tilde{\Delta}\rangle$', r'$\sigma_{x}/\langle x\rangle$', r' Noisy integrator with $\sigma_{s}/\langle\tilde{\Delta}\rangle=$', r'$\sigma_{s}/\langle\tilde{\Delta}\rangle=$']
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent [Whi5]$<1$ at division', 'Percent [Whi5]$<1$ at birth']
celltype = ['Mothers', 'Daughters']
sim_type = ['Discr gen ', 'Discr time ']

r_nums = [0, 1]
cmap_lims=[1.0, 1.4]

# tester that simulations populated entire matrix:
print np.count_nonzero(data == 0)
temp = data[:,:,:,:,:,:,:,0]
print np.count_nonzero(temp == 0)
del temp


for i0 in range(len(g1_thresh_std)):
    for i1 in range(len(r_nums)):
        for i2 in range(num_meas):
            i3 = 1
            i4=0  # Mothers
            plotlab = ['(A) ', '(B) ', '(C) ', '(D) ']
            fig = plt.figure(figsize=[8, 8])
            # plots for the paper. Daughters only. Individual plots so can be mixed and matched.
            # Discretized time only.
            ax = plt.subplot(1, 1, 1)
            obs = np.mean(data[r_nums[i1], :, i0, :, :, i4, i3, i2], axis=-1)
            temp = plotlab[2*np.mod(i0,2)+i1]+labels[3] + str(g1_thresh_std[i0]) + r', $\langle x\rangle=$' + str(r[r_nums[i1]])
            if i2 == 0:
                ax = g.heat_map_pd(obs, r_std, d_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
                                color='black', cmap_lims=cmap_lims)
                fig.savefig(
                    './1705_inh_yeast_asymm_fig_{0}_meas_{1}_si_{2}_r_{3}_sim_{4}_celltype_{5}_v2.eps'.format(
                        str(par1['modeltype']), str(i2), str(i0), str(r_nums[i1]), str(i3), str(i4)),
                    bbox_inches='tight', dpi=fig.dpi)
                del fig, obs, ax
            i4=1  #Daughters
            plotlab = ['(C) ', '(D) ', '(E) ', '(F) ']
            fig = plt.figure(figsize=[8, 8])
            # plots for the paper. Daughters only. Individual plots so can be mixed and matched.
            # Discretized time only.
            ax = plt.subplot(1, 1, 1)
            obs = np.mean(data[r_nums[i1], :, i0, :, :, i4, i3, i2], axis=-1)
            temp = plotlab[2*np.mod(i0,2)+i1]+labels[3] + str(g1_thresh_std[i0]) + r', $\langle x\rangle=$' + str(r[r_nums[i1]])
            if i2 == 0:
                ax = g.heat_map_pd(obs, r_std, d_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
                                color='black', cmap_lims=cmap_lims)
                fig.savefig(
                    './1705_inh_yeast_asymm_fig_{0}_meas_{1}_si_{2}_r_{3}_sim_{4}_celltype_{5}_v2.eps'.format(
                        str(par1['modeltype']), str(i2), str(i0), str(r_nums[i1]), str(i3), str(i4)),
                    bbox_inches='tight', dpi=fig.dpi)
                del fig, obs, ax


            # else:
            #     ax == g.heat_map(obs, r_std, d_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
            #                     fmt='.3g')

            # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
            # will be modified.
