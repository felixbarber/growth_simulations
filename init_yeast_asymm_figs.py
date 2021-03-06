#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import growth_simulation_accumulator_asymmetric as h
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import pandas as pd
import seaborn as sns
# Setting SNS styles
sns.set_context('paper', font_scale=2.2)
sns.set_style('ticks')

par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3), ('g2_std', 0.02), ('l_std', 0.02),
            ('CD', 0.65)])

# Now we create the heatmaps that will be used in the paper for the asymmetric yeast section.
filenames = ['init_yeast_asymm_complete.npy']
data=np.load(filenames[0])
print data.shape
delta = 10.0
par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3)])
L = 31
r = np.linspace(0.5, 0.7, 2)
r_std = np.linspace(0.0, 0.3, L)
g1_thresh_std = np.linspace(0.0, 0.3, L)

# models = [3, 4]
num_celltype = 2
num_sims = 2
num_meas = 3
num_rep = 50
timer = 7.6
celltype=['Mother', 'Daughter']

# X = [len(r), len(r_std), len(g1_thresh_std), num_rep, num_celltype, num_sims, num_meas]
# celltype = ['Mothers', 'Daughters', 'Population']
model_descr = ['Cells can shrink', 'Cells cannot shrink']
obs_type = ['$V_b$ $V_d$ slope', '% with $A_b><A_c>$']
lab = [r'$\sigma_i/\langle A_c\rangle$', r'$\sigma_{x}/\langle x\rangle$']
r_vals = [0, 1]
label = ['(A) ', '(B) ', '(C) ', '(D) ']
cmap_lims = [0.7, 1.3]
for i1 in range(num_celltype):
    # for i2 in range(num_sims):
    i2 = 1  # discretized time. num_sims
    i3 = 0  # slope only. num_meas
        # for i3 in range(num_meas):
    for i4 in range(len(r_vals)):
        obs = np.mean(data[r_vals[i4], :, :, :, i1, i2, i3], axis=-1)
        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1, 1, 1)
        ax = plt.imshow(np.absolute(obs - 1.0) > 0.1)
        fig.savefig('./temp_{0}_{1}_{2}_{3}'.format(str(i1), str(i3),
                                                    str(i2), str(r_vals[i4])))
        del fig
        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1, 1, 1)
        temp = label[2*(1-i1)+i4] + celltype[i1]+', '+r' $\langle x\rangle=${0}'.format(np.round(r[r_vals[i4]], 2))
        print i1, i3, i2, i4
        ax = g.heat_map_pd(obs, r_std, g1_thresh_std, ax, xlabel=lab[0], ylabel=lab[1], title=temp,
                           cmap_lims=cmap_lims)
        #ax = g.heat_map(obs, r_std, g1_thresh_std, ax, xlabel=lab[0], ylabel=lab[1], title=temp)
        fig.savefig('./1705_init_yeast_asymm_celltype_{0}_meas_{1}_sim_{2}_r_{3}_v1.eps'.format(str(i1), str(i3),
                                                        str(i2), str(r_vals[i4])), bbox_inches='tight', dpi=fig.dpi)
        del fig

# April17_paper_scripts_4_model_{0}_celltype_{1}_meas_{2}_sim_{3}_r_{4}.eps allows negative growth through noise, but not
# in G1. Comes from March17_accumulator1_V0.npy
# April17_paper_scripts_4_model_{0}_celltype_{1}_meas_{2}_sim_{3}_r_{4}.eps has no negative growth whatsoever in models
# 2 and 4. comes from April17_paper_scripts_5_v0.
