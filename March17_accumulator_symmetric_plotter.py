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

data = np.load('March17_accumulator_symmetric_V0.npy')

delta = 10.0

par_vals = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
                ('r', 1.0), ('r_std', 0.0), ('delta', 10.0), ('g1_thresh_std', 0.0)])

L = 7
models = [3, 4]
g1_thresh_std = np.linspace(0.0, 0.3, L)
r_std = np.linspace(0.0, 0.3, L)

vals = [['g1_thresh_std', 'r_std'], ['g1_thresh_std', 'r_std']]
pars = [[g1_thresh_std, r_std], [g1_thresh_std, r_std]]
num_rep = 5  # number of repeats for each condition
num_celltype = 2
num_sims = 2  # number of different simulations to be run
num_meas = 2  # number of different kinds of measurements
models_descr = ['(A)', '(B)']
# X = [len(models), L, L, num_rep, num_celltype, num_sims, num_meas]
labels = [['$\sigma_{r}/r$', '$\sigma_{\Delta}/\Delta$'], ['$\sigma_{r}/r$', '$\sigma_{\Delta}/\Delta$']]
# format is [xlabel, ylabel]
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent $A>\Delta$ at birth']
celltype = ['Mothers', 'Daughters']
sim_type = [' Discr gen ', ' Discr time ']
for i0 in range(len(models)):
    for i2 in range(num_meas):
        for i3 in range(num_sims):
            fig = plt.figure(figsize=[16, 8])
            for i4 in range(num_celltype):
                ax = plt.subplot(1, 2, 1+i4)
                obs = np.mean(data[i0, :, :, :, i4, i3, i2], axis=-1)
                ax = g.heat_map(obs, pars[i0][0], pars[i0][1], ax, xlabel=labels[i0][0], ylabel=labels[i0][1],
                                title=celltype[i4])
            temp = models_descr[i0]+sim_type[i3]+meas_type[i2]
            plt.suptitle(temp)
            fig.savefig('./March17_accumulator_symmetric_plotter_model_'+str(models[i0])+'_meas_'+str(i2)+'_sim_'+str(i3)+
                        '.eps', bbox_inches='tight', dpi=fig.dpi)
            del fig
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.

# Now we do the plots for the paper
for i0 in range(len(models)):
    for i2 in range(num_meas):
        i3 = 1  # discretized time only.
        for i4 in range(num_celltype):
            fig = plt.figure(figsize=[8, 8])
            ax = plt.subplot(1, 1, 1)
            obs = np.mean(data[i0, :, :, :, i4, i3, i2], axis=-1)
            temp = models_descr[i0]
            ax = g.heat_map(obs, pars[i0][0], pars[i0][1], ax, xlabel=labels[i0][0], ylabel=labels[i0][1],
                            title=temp)
            fig.savefig('./March17_accumulator_symmetric_plotter_model_'+str(models[i0])+'_meas_'+str(i2)+'_sim_'+str(i3)
                        +'celltype'+str(i4)+'.eps', bbox_inches='tight', dpi=fig.dpi)
            del fig

temp = ['Shrinking allowed', 'No shrinking']
for i4 in range(num_celltype):
    fig = plt.figure(figsize=[8, 8])
    for i0 in range(len(models)):
        i2 = 0  # slope only
        i3 = 1  # discretized time only.
        # i4 = 1  # daughter cells only.

        obs = np.mean(data[i0, :, 0, :, i4, i3, i2], axis=-1)
        # print obs.shape
        plt.plot(g1_thresh_std, obs, label=temp[i0])
    plt.title(models_descr[i4], size=20)
    plt.legend()
    plt.xlabel(labels[0][1], size=20)
    plt.ylabel(meas_type[0], size=20)
    plt.xticks(size=14)
    plt.yticks(size=14)
    # plt.show()
    fig.savefig('./March17_accumulator_symmetric_plotter_lines'+'celltype'+str(i4)+'.eps', bbox_inches='tight',
                    dpi=fig.dpi)
    del fig
