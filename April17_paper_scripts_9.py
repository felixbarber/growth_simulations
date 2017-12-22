#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

plt.rc('text', usetex=True)
font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

data = np.load('March17_accumulator_symmetric_V1.npy')

delta = 10.0

par_vals = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
                ('r', 1.0), ('r_std', 0.0), ('delta', 10.0), ('g1_thresh_std', 0.0)])

L = 13
models = [3, 4]
g1_thresh_std = np.linspace(0.0, 0.3, L)
r_std = np.linspace(0.0, 0.3, L)
print data.shape

vals = [['g1_thresh_std', 'r_std'], ['g1_thresh_std', 'r_std']]
pars = [[g1_thresh_std, r_std], [g1_thresh_std, r_std]]
num_rep = 20  # number of repeats for each condition
num_celltype = 3
num_sims = 2  # number of different simulations to be run
num_meas = 2  # number of different kinds of measurements
models_descr = ['(A)', '(B)']
# X = [len(models), L, L, num_rep, num_celltype, num_sims, num_meas]
labels = [['$\sigma_{r}/r$', r'$\sigma_{\Delta}/\langle A_c\rangle$'], ['$\sigma_{r}/r$', r'$\sigma_{\Delta}/\langle A_c\rangle$']]
# format is [xlabel, ylabel]
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent $A>\Delta$ at birth']
sim_type = [' Discr gen ', ' Discr time ']

temp = ['Shrinking allowed', 'No shrinking']
lab = ['(A)', '(B)']
celltype = ['Mothers', 'Daughters']
loc = [2, 1]
colors = ['cornflowerblue', 'salmon']
for i0 in range(len(models)):
    fig = plt.figure(figsize=[8, 8])
    for i4 in range(2):
        i2 = 0  # slope only
        i3 = 1  # discretized time only.
        # i4 = 1  # daughter cells only.
        obs = np.mean(data[i0, :, 0, :, i4, i3, i2], axis=-1)
        stds = np.std(data[i0, :, 0, :, i4, i3, i2], axis=-1)
        # print obs.shape
        plt.plot(g1_thresh_std, obs, colors[i4], label=celltype[i4]+' mean')
        plt.fill_between(g1_thresh_std, obs - stds, obs + stds, facecolor=colors[i4], alpha=0.2)
    plt.title(lab[i0], size=20)
    plt.legend(loc=loc[i0])
    plt.xlabel(labels[0][1], size=20)
    plt.ylabel(meas_type[0], size=20)
    plt.xticks(size=14)
    plt.yticks(size=14)
    # plt.show()
    fig.savefig('./April17_paper_scripts_9_model_{0}.png'.format(models[i0]), bbox_inches='tight', dpi=fig.dpi)
    del fig

model_descr = ['A', 'No shrinking']

fig = plt.figure(figsize=[8, 8])
for i0 in range(len(models)):
    i4 = 2  # population level only
    i2 = 0  # slope only
    i3 = 1  # discretized time only.
    # i4 = 1  # daughter cells only.
    # print data[i0, 0, 0, :, i4, i3, i2]
    obs = np.mean(data[i0, :, 0, :, i4, i3, i2], axis=-1)
    stds = np.std(data[i0, :, 0, :, i4, i3, i2], axis=-1)
    # print obs.shape
    plt.plot(g1_thresh_std, obs, colors[i0], label=model_descr[i0])
    plt.fill_between(g1_thresh_std, obs - stds, obs + stds, facecolor=colors[i0], alpha=0.2)
    # plt.title(lab[i0], size=20)
    plt.legend(loc=loc[i0])
    plt.xlabel(labels[0][1], size=20)
    plt.ylabel(meas_type[0], size=20)
    plt.ylim(ymin=0.8, ymax=2.2)
    plt.xticks(size=14)
    plt.yticks(size=14)
    # plt.show()
fig.savefig('./April17_paper_scripts_9_fullpop.png'.format(models[i0]), bbox_inches='tight', dpi=fig.dpi)
del fig

model_descr = ['A: ', 'B: ', 'C: ', 'D: ']

g1_thresh_std_ints = [0, 8]
colors = ['cornflowerblue', 'salmon', 'palegreen', 'sandybrown']
fig = plt.figure(figsize=[8, 8])
ax=plt.subplot(1,1,1)
cind = 0
xlims=[0.0, 0.32]
ylims=[0.5, 2.1]
for i0 in range(len(models)):
    for i1 in range(len(g1_thresh_std_ints)):
        i4 = 2  # population level only
        i2 = 0  # slope only
        i3 = 1  # discretized time only.
        # i4 = 1  # daughter cells only.
        # print data[i0, 0, 0, :, i4, i3, i2]
        obs = np.mean(data[i0, i1, :, :, i4, i3, i2], axis=-1)
        stds = np.std(data[i0, i1, :, :, i4, i3, i2], axis=-1)
        # print obs.shape
        plt.plot(r_std, obs, colors[cind],
                 label=model_descr[2*i0+i1]+r' $\sigma_i/\langle A_c\rangle=${0}'.format(g1_thresh_std[g1_thresh_std_ints[i1]]))
        plt.fill_between(r_std, obs - stds, obs + stds, facecolor=colors[cind], alpha=0.2)
        # plt.title(lab[i0], size=20)
        plt.legend(loc=7)
        plt.xlabel('$\sigma_r/r$', size=20)
        plt.ylabel(meas_type[0], size=20)
        plt.ylim(ymin=0.5, ymax=2.1)

        plt.xlim(xmin=xlims[0], xmax=xlims[1])
        plt.ylim(ymin=ylims[0], ymax=ylims[1])

        ax.set_facecolor('white')
        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='x', colors='black', **tkw)
        ax.tick_params(axis='y', colors='black', **tkw)
        plt.axhline(ylims[0], color='black')
        plt.axvline(xlims[0], color='black')
        plt.axhline(ylims[1], color='black')
        plt.axvline(xlims[1], color='black')
        plt.xticks(size=18)
        plt.yticks(size=18)

        # plt.show()
        cind += 1
fig.savefig('./April17_paper_scripts_9_fullpop_rstd.eps'.format(models[i0]), bbox_inches='tight', dpi=fig.dpi)
del fig