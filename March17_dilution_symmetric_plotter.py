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

data = np.load('March17_dilution_symmetric_V0.npy')

delta = 10.0

models_descr = ['Noisy synthesis rate', 'Noisy synthesis rate NNG', 'Noisy integrator', 'Noisy integrator NNG',
                'Fixed r noisy integrator', 'Fixed r noisy integrator NNG']

L = 7
models = [5, 10, 17, 18, 23, 24]
g1_thresh_std = np.linspace(0.0, 0.3, L)
k_std = np.linspace(0.04, 0.3, L)
d_std = np.linspace(0.04, 0.3, L)
vals = [['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std'],
        ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std']]
pars = [[g1_thresh_std, k_std], [g1_thresh_std, k_std], [g1_thresh_std, d_std], [g1_thresh_std, d_std],
        [g1_thresh_std, d_std], [g1_thresh_std, d_std]]
num_rep = 3  # number of repeats for each condition
num_celltype = 2
num_sims = 2  # number of different simulations to be run
num_meas = 3  # number of different kinds of measurements
# X = [len(models), L, L, num_rep, num_celltype, num_sims, num_meas]
labels = [['$\sigma_{K}/K$', '$\sigma_{i}/\Delta$'], ['$\sigma_{\Delta}/<\Delta>$', '$\sigma_{i}/<\Delta>$'],
          ['$\sigma_{\Delta}/\Delta$', '$\sigma_{i}/\Delta$']]  # format is [xlabel, ylabel]
meas_type = ['$V_{b}$ $V_{d}$ slope', 'Percent [Whi5]$<1$ at division', 'Percent [Whi5]$<1$ at birth']
celltype = ['Mothers', 'Daughters']
sim_type = [' Discr gen ', ' Discr time ']
models_descr_full = ['(A)', '(B)']
itemp=0  # indexes the plots for the paper.
for i0 in range(len(models)):
    for i2 in range(num_meas):
        for i3 in range(num_sims):
            fig = plt.figure(figsize=[16, 8])
            for i4 in range(num_celltype):
                ax = plt.subplot(1, 2, 1+i4)
                obs = np.mean(data[i0, :, :, :, i4, i3, i2], axis=-1)
                ax = g.heat_map(obs, pars[i0][0], pars[i0][1], ax, xlabel=labels[i0/2][0], ylabel=labels[i0/2][1],
                                title=celltype[i4])
            temp = models_descr[i0]+sim_type[i3]+meas_type[i2]
            plt.suptitle(temp)
            fig.savefig('./March17_dilution_symmetric_plotter_model_'+str(models[i0])+'_meas_'+str(i2)+'_sim_'+str(i3)+
                        '.eps', bbox_inches='tight', dpi=fig.dpi)
            del fig
            # plots for the paper. duaghters only, with discretized time.
    if models[i0] in [17, 18]:
        for i2 in range(num_meas):
            i3 = 1  # discretized time simulations
            fig = plt.figure(figsize=[16, 8])
            i4 = 1  # daughters only
            ax = plt.subplot(1, 2, 1 + i4)
            obs = np.mean(data[i0, :, :, :, i4, i3, i2], axis=-1)
            temp = models_descr_full[itemp]
            if models[i0] == 17:
                fmt = '.1g'
            else:
                fmt = '.2g'
            ax = g.heat_map(obs, pars[i0][0], pars[i0][1], ax, xlabel=labels[i0 / 2][0],
                ylabel=labels[i0 / 2][1], title=temp, fmt=fmt)
            fig.savefig('./March17_dilution_symmetric_plotter_D_model_' + str(models[i0]) + '_meas_' + str(
                i2) + '_sim_' + str(i3) +
                        '.eps', bbox_inches='tight', dpi=fig.dpi)
            del fig
        itemp += 1
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.

# Now we do plotting for March17_dilution_symmetric_3.py
data = np.load('March17_dilution_symmetric_3.npy')
td = 1.0
delta = 10.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('CD', 1.0),
                 ('g2_std', 0.2), ('d_std', 0.1), ('g1_thresh_std', 0.1)])
num_rep = 4
N = 50001
# N=10
save_freq = 1000
X = [num_rep, N, 2, 8]
# temp2 = np.asarray([obj.vb for obj in temp if obj.isdaughter == i2 and obj.exists])
# temp3 = np.asarray([obj.wb for obj in temp if obj.isdaughter == i2 and obj.exists])
# a[i0, i1, i2, 0] = np.mean(temp2)
# a[i0, i1, i2, 1] = np.std(temp2)
# a[i0, i1, i2, 2] = np.mean(temp3)
# a[i0, i1, i2, 3] = np.std(temp3)
# a[i0, i1, i2, 4] = np.mean(np.log(temp2))
# a[i0, i1, i2, 5] = np.std(np.log(temp2))
# a[i0, i1, i2, 6] = np.mean(np.log(temp3))
# a[i0, i1, i2, 7] = np.std(np.log(temp3))

tvec = np.linspace(1, N, N)*par_vals['nstep']*par_vals['dt']
fig = plt.figure(figsize=[6, 6])
for i0 in range(2):
    for i1 in range(data.shape[0]):
        plt.plot(tvec[::1000], data[i1, ::1000, i0, 7]/data[i1, ::1000, i0, 7], label=celltype[i0]+' rep {0}'.format(i1))
plt.xlabel('Time')
plt.title('CV($V_b$) vs. time for a symmetrically dividing population')
fig.savefig('./March17_dilution_symmetric_4_model_{0}.eps'.format(par_vals['modeltype']), bbox_inches='tight', dpi=fig.dpi)
