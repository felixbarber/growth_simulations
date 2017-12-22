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
             ('num_gen', 9), ('modeltype', 24), ('delta', delta)])

# X = [len(r), len(r_std), len(g1_thresh_std), len(d_std), num_rep, num_celltype, num_sims, num_meas]

model_descr = ['Noisy integrator no neg growth']
# r = np.linspace(0.45, 1.0, 12)
r = np.linspace(0.45, 1.0, 12)
# r_std = np.linspace(0.0, 0.28, 8)
r_std = np.linspace(0.0, 0.3, 7)
# d_std = np.linspace(0.0, 0.28, 8)
d_std = np.linspace(0.0, 0.3, 7)
g1_thresh_std = np.linspace(0.0, 0.3, 4)
num_rep = 3
num_celltype = 2
num_sims = 2
num_meas = 3
# filename = 'April17_paper_scripts_2_model' + str(par1['modeltype']) + '_V1'
filename = 'April17_paper_scripts_2_model' + str(par1['modeltype']) + '_V2'
data = np.load(filename + '.npy')

labels = [r'$\sigma_{\Delta}/\langle\Delta\rangle$', '$\sigma_{r}/r$', r' Noisy integrator with $\sigma_{i}/\langle\Delta\rangle=$', r'$\sigma_{i}/\langle\Delta\rangle=$']
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent [Whi5]$<1$ at division', 'Percent [Whi5]$<1$ at birth']
celltype = ['Mothers', 'Daughters']
sim_type = ['Discr gen ', 'Discr time ']
plotlab = ['(C) ', '(D) ', '(E) ', '(F) ']
r_nums = [1, 5]
cmap_lims=[1.0, 1.4]
# for i0 in range(len(g1_thresh_std)):
#     for i1 in range(len(r_nums)):
#         for i2 in range(num_meas):
#             i3 = 1
#             for i4 in range(2): # mothers and daughters
#                 fig = plt.figure(figsize=[8, 8])
#                 # plots for the paper. Daughters only. Individual plots so can be mixed and matched.
#                 # Discretized time only.
#                 ax = plt.subplot(1, 1, 1)
#                 obs = np.mean(data[r_nums[i1], :, i0, :, :, i4, i3, i2], axis=-1)
#                 temp = plotlab[2*np.mod(i0,2)+i1]+labels[3] + str(g1_thresh_std[i0]) + ', $r=$' + str(r[r_nums[i1]])
#                 if i2 == 0:
#                     ax = g.heat_map_1(obs, r_std, d_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
#                                     color='black', cmap_lims=cmap_lims)
#                 else:
#                     ax == g.heat_map(obs, r_std, d_std, ax, xlabel=labels[0], ylabel=labels[1], title=temp,
#                                     fmt='.3g')
#                 fig.savefig(
#                     './April17_paper_scripts_1_model_{0}_meas_{1}_si_{2}_r_{3}_sim_{4}_celltype_{5}_v2.eps'.format(
#                         str(par1['modeltype']), str(i2), str(i0), str(r_nums[i1]), str(i3), str(i4)), bbox_inches='tight', dpi=fig.dpi)
#                 del fig, obs, ax
#                 # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
#                 # will be modified.

# this part generates the distribution subfigures

plotlab = ['(A) ', '(B) ']
par1 = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
             ('num_gen', 9), ('modeltype', 24), ('delta', delta), ('r', 0.5), ('d_std', 0.05), ('g1_thresh_std', 0.05)])
r_std = np.linspace(0.0, 0.2, 2)
num_bins = 20
l = 3
# for i0 in range(len(r_std)):
#     par1['r_std'] = r_std[i0]
#     if i0 == 0:
#         print par1
#     print 'r_std:', par1['r_std']
#     temp1 = g.discr_gen(par1)
#     # This will initialize the subsequent simulations for this model
#     temp2 = g.starting_popn_seeded([obj for obj in temp1 if obj.exists], par1)
#     # initial pop seeded from c
#     temp3, obs3 = g.discr_time_1(par1, temp2)
#     for i1 in range(2):
#         fig = plt.figure(figsize=[6, 6])
#         ax = plt.subplot(1, 1, 1)
#         plt.xlabel('$V_b$')
#         plt.ylabel('$V_d$')
#         plt.title(plotlab[i0])
#         x = np.asarray([obj.vb for obj in temp3 if obj.exists and obj.isdaughter == i1])
#         y = np.asarray([obj.vd for obj in temp3 if obj.exists and obj.isdaughter == i1])
#
#         plt.hexbin(x, y, cmap="Purples", gridsize=60)
#         plt.ylim(ymin=np.mean(y) - l * np.std(y), ymax=np.mean(y) + l * np.std(y))
#         plt.xlim(xmin=np.mean(x) - l * np.std(x), xmax=np.mean(x) + l * np.std(x))
#         bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, np.arange(len(x)), statistic='mean',
#                                                                        bins=num_bins, range=None)
#         bin_posns = list([])
#         y_av = list([])
#         y_sem = list([])
#         bin_err = list([])
#         for j in range(num_bins):
#             bin_no = binnumber == j + 1
#             if np.sum(bin_no) > 20:  # we don't want to be skewed by having too few data points in a fit
#                 y_av.append(np.mean(y[np.nonzero(bin_no)]))
#                 y_sem.append(np.std(y[np.nonzero(bin_no)]) / np.sqrt(np.sum(bin_no)))
#                 bin_posns.append((bin_edges[j] + bin_edges[j + 1]) / 2)
#                 bin_err.append((bin_edges[j + 1] - bin_edges[j]) / 2)
#         y_av = np.asarray(y_av)
#         y_sem = np.asarray(y_sem)
#         bin_posns = np.asarray(bin_posns)
#         bin_err = np.asarray(bin_err)
#         plt.errorbar(bin_posns, y_av, yerr=y_sem, label="binned means", ls="none", color="r")
#
#         vals = scipy.stats.linregress(x, y)
#         x1 = np.linspace(np.mean(x) - 2*np.std(x), np.mean(x)+2*np.std(x), 10)
#         y1 = vals[0]*x1+vals[1]
#         plt.plot(x1, y1, label=r'Linear regression slope $=${0}'.format(np.round(vals[0], 2)))
#         plt.legend(loc=2)
#
#         tkw = dict(size=4, width=1.5)
#         ax.set_facecolor('white')
#         ax.tick_params(axis='x', colors='black', **tkw)
#         ax.tick_params(axis='y', colors='black', **tkw)
#         tmp = [ax.get_ylim(), ax.get_xlim()]
#         plt.axhline(tmp[0][0], color='black')
#         plt.axvline(tmp[1][0], color='black')
#         plt.axhline(tmp[0][1], color='black')
#         plt.axvline(tmp[1][1], color='black')
#         plt.xticks(size=18)
#         plt.yticks(size=18)
#         fig.savefig('./April17_paper_scripts_1_dist_model_{0}_noise_{1}_celltype_{2}.jpg'.format(
#                         str(par1['modeltype']), str(i0), str(i1)), bbox_inches='tight', dpi=fig.dpi)

for i1 in range(2):
    fig = plt.figure(figsize=[14, 6])
    for i0 in range(len(r_std)):
        ax = plt.subplot(1, 2, 1+i0)
        par1['r_std'] = r_std[i0]
        if i0 == 0:
            print par1
        print 'r_std:', par1['r_std']
        temp1 = g.discr_gen(par1)
        # This will initialize the subsequent simulations for this model
        temp2 = g.starting_popn_seeded([obj for obj in temp1 if obj.exists], par1)
        # initial pop seeded from c
        temp3, obs3 = g.discr_time_1(par1, temp2)
        plt.xlabel('$V_b$')
        plt.ylabel('$V_d$')
        # plt.title(plotlab[i0])
        x = np.asarray([obj.vb for obj in temp3 if obj.exists and obj.isdaughter == i1])
        y = np.asarray([obj.vd for obj in temp3 if obj.exists and obj.isdaughter == i1])

        plt.hexbin(x, y, cmap="Purples", gridsize=40)
        plt.ylim(ymin=np.mean(y) - l * np.std(y), ymax=np.mean(y) + l * np.std(y))
        plt.xlim(xmin=np.mean(x) - l * np.std(x), xmax=np.mean(x) + l * np.std(x))
        bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, np.arange(len(x)), statistic='mean',
                                                                       bins=num_bins, range=None)
        bin_posns = list([])
        y_av = list([])
        y_sem = list([])
        bin_err = list([])
        for j in range(num_bins):
            bin_no = binnumber == j + 1
            if np.sum(bin_no) > 20:  # we don't want to be skewed by having too few data points in a fit
                y_av.append(np.mean(y[np.nonzero(bin_no)]))
                y_sem.append(np.std(y[np.nonzero(bin_no)]) / np.sqrt(np.sum(bin_no)))
                bin_posns.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                bin_err.append((bin_edges[j + 1] - bin_edges[j]) / 2)
        y_av = np.asarray(y_av)
        y_sem = np.asarray(y_sem)
        bin_posns = np.asarray(bin_posns)
        bin_err = np.asarray(bin_err)
        (_, caps, _) = plt.errorbar(bin_posns, y_av, yerr=y_sem, label="binned means", ls="none", color="r", elinewidth=3, capsize=4)

        for cap in caps:
            cap.set_color('red')
            cap.set_markeredgewidth(5)

        vals = scipy.stats.linregress(x, y)
        x1 = np.linspace(np.mean(x) - 2*np.std(x), np.mean(x)+2*np.std(x), 10)
        y1 = vals[0]*x1+vals[1]
        plt.plot(x1, y1, label=r'Linear regression slope $=${0}'.format(np.round(vals[0], 2)))
        plt.legend(loc=2)
        plt.xlim(xmin=0)

        tkw = dict(size=4, width=1.5)
        ax.set_facecolor('white')
        ax.tick_params(axis='x', colors='black', **tkw)
        ax.tick_params(axis='y', colors='black', **tkw)
        tmp = [ax.get_ylim(), ax.get_xlim()]
        plt.axhline(tmp[0][0], color='black')
        plt.axvline(tmp[1][0], color='black')
        plt.axhline(tmp[0][1], color='black')
        plt.axvline(tmp[1][1], color='black')
        plt.xticks(size=18)
        plt.yticks(size=18)
    fig.savefig('./April17_paper_scripts_1_dist_model_{0}_celltype_{1}.eps'.format(
                        str(par1['modeltype']), str(i1)), dpi=fig.dpi, bbox_inches='tight')
    del fig
