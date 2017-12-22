#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
from matplotlib.font_manager import FontProperties
from matplotlib import rc

plt.rc('text', usetex=True)
font = {'family': 'normal', 'weight': 'bold', 'size': 16}
plt.rc('font', **font)

# fontP = FontProperties()
# fontP.set_size('small')

# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

# data = np.load('March17_dilution_symmetric_V1.npy')
data = np.load('April17_paper_scripts_10_V0.npy')
delta = 10.0

models_descr = ['Noisy synthesis rate', 'Noisy synthesis rate NNG', 'Noisy integrator', 'Noisy integrator NNG',
                'Fixed r noisy integrator', 'Fixed r noisy integrator NNG']

# L = 7
# models = [5, 10, 17, 18, 23, 24]
# g1_thresh_std = np.linspace(0.0, 0.3, L)
# k_std = np.linspace(0.04, 0.3, L)
# d_std = np.linspace(0.04, 0.3, L)
# vals = [['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std'],
#         ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std']]
# pars = [[g1_thresh_std, k_std], [g1_thresh_std, k_std], [g1_thresh_std, d_std], [g1_thresh_std, d_std],
#         [g1_thresh_std, d_std], [g1_thresh_std, d_std]]
# num_rep = 20  # number of repeats for each condition
# num_celltype = 2
# num_sims = 2  # number of different simulations to be run
# num_meas = 3  # number of different kinds of measurements
# labels = [['$\sigma_{K}/K$', '$\sigma_{i}/\Delta$'], ['$\sigma_{\Delta}/<\Delta>$', '$\sigma_{i}/<\Delta>$'],
#           ['$\sigma_{\Delta}/\Delta$', '$\sigma_{i}/\Delta$']]  # format is [xlabel, ylabel]

labels = [r'$\sigma_{\Delta}/\langle A_c\rangle$', r'$\sigma_{i}/\langle A_c\rangle$']  # X label and Y label

L = 7
models = [23, 24]
g1_thresh_std = np.linspace(0.0, 0.3, L)
d_std = np.linspace(0.0, 0.3, L)
r_std = np.linspace(0.0, 0.3, L)
vals = ['g1_thresh_std', 'd_std', 'r_std']
pars = [g1_thresh_std, d_std, r_std]
num_rep = 20  # number of repeats for each condition
num_celltype = 3
num_sims = 2  # number of different simulations to be runAA
num_meas = 3  # number of different kinds of measurements

# X = [len(models), L, L, L, num_rep, num_celltype, num_sims, num_meas]

meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent [Whi5]$<1$ at division', 'Percent [Whi5]$<1$ at birth']
celltype = ['Mothers', 'Daughters', 'Population']
sim_type = [' Discr gen ', ' Discr time ']
models_descr_full = ['(A)', '(B)']
i2, i3 = 1, 0  # discretized time, and slope measurement
cmap_lims = [0.0, 2.0]
print data.shape
# for i0 in range(len(models)):
#     for i4 in range(num_celltype):
#         fig = plt.figure(figsize=[8, 8])
#         ax = plt.subplot(1, 1, 1)
#         obs = np.mean(data[i0, :, :, 0, :, i4, i2, i3], axis=-1)  # r_std = 0 only
#         # if np.mod(i0, 2) == 0:
#         #     temp = 1.0
#         #     temp1 = 0.1
#         #     out = True
#         #     alpha = 1.0
#         # else:
#         #     temp = 2.0
#         #     temp1 = 0.05
#         #     out = True
#         #     alpha = 0.0
#         # ax = g.heat_map_1(obs, pars[i0][0], pars[i0][1], ax, xlabel=labels[i0/2][0], ylabel=labels[i0/2][1],
#         #                 title=models_descr_full[np.mod(i0, len(models_descr_full))], bound=temp1, val=temp, outline=out,
#         #                   alpha=alpha, cmap_lims=cmap_lims)
#         ax = g.heat_map_1(obs, pars[0], pars[1], ax, xlabel=labels[0], ylabel=labels[1],
#                           title=models_descr_full[np.mod(i0, len(models_descr_full))], cmap_lims=cmap_lims)
#         fig.savefig('./April_paper_scripts_3_model_{0}_celltype_{1}_sim_{2}_meas_{3}_V1.eps'.format(str(models[i0]),
#                     str(i4), str(i2), str(i3)), bbox_inches='tight', dpi=fig.dpi)
#         del fig

inds = [0, 6]
i3, i4, i5 = 2, 1, 0  # population level, discretized time, slope.
model_descr = ['Shrinking', 'No shrinking']
lims = [[0.0, 2.1], [0.0, 2.1]]
loc = [1, 5]
titles = ['(A)', '(B)']
# colors = ['cornflowerblue', 'salmon', 'palegreen', 'sandybrown']
xlims=[0.0, 0.32]
for i0 in range(len(models)):
    fig = plt.figure(figsize=[8, 8])
    ax = plt.subplot(1, 1, 1)
    for i1 in inds:  # sigma_i
        for i2 in inds:  # d_std
            obs = np.mean(data[i0, i1, i2, :, :, i3, i4, i5], axis=-1)
            stds = np.std(data[i0, i1, i2, :, :, i3, i4, i5], axis=-1)
            plt.plot(r_std, obs, label=r'$\sigma_i/\langle A_c\rangle=${0}, $\sigma_\Delta/\langle A_c\rangle=${1}'.format(str(g1_thresh_std[i1]), str(d_std[i2])))
            plt.fill_between(r_std, obs - stds, obs + stds, alpha=0.2)
            # plt.title(lab[i0], size=20)
    plt.legend(loc=loc[i0])

    plt.xlabel(r'$\sigma_r/\langle r\rangle$', size=20)
    plt.ylabel(meas_type[0], size=20)
    # plt.title(titles[i0], size=20)

    plt.xlim(xmin=xlims[0], xmax=xlims[1])
    plt.ylim(ymin=lims[i0][0], ymax=lims[i0][1])
    ax.set_facecolor('white')
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', colors='black', **tkw)
    ax.tick_params(axis='y', colors='black', **tkw)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axhline(lims[i0][1], color='black')
    plt.axvline(xlims[1], color='black')
    plt.xticks(size=18)
    plt.yticks(size=18)

    print ax.get_ylim(), ax.get_xlim()

    plt.show(fig)
    fig.savefig(
        './April17_paper_scripts_3_celltype_{0}_sim_{1}_meas_{2}_model_{3}.png'.format(str(i3), str(i4), str(i5), str(models[i0])), dpi=fig.dpi)
    del fig

# V0 for allowing negative growth through noise.
# V1 for not allowing negative growth.
