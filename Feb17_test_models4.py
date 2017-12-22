#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy


data = np.load('Feb17_test_models3_model16.npy')
print data.shape
d = np.mean(data, axis=7)
celltype = ['m', 'd']
sims = ['Generation', 'Time', 'Theory']
# i6 = 1
vals = np.zeros([5, 2, 4])
# X = [num_med, len_cd, len(g1_std), len_var, len_var, len(k_std), len_var, num_rep, num_celltype, num_sim]
# vals = ['placeholder', 'CD', 'g1_thresh_std', 'g2_std', 'l_std', 'k_std', 'w_frac', num_rep, num_celltype, num_sim]
i0 = 0
i2 = 1
inds = (np.asarray(data.shape)[1:7]-1)/2
# for i1 in range(data.shape[8]):
#     temp = np.mean(data[i0, :3, :, :2, :, :, :, :, i1, i2], axis=6)
#     vals[i0, i1, 0] = np.amin(temp)
#     vals[i0, i1, 1] = np.amax(temp)
#     vals[i0, i1, 2] = np.mean(data[i0, inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], :, i1, i2])
#

for i0 in range(data.shape[0]):  # medium. Deal with glucose separately.
    for i1 in range(data.shape[8]):  # celltype
        for i2 in range(data.shape[9]):  # simulation type
            # temp = np.mean(data[i0, :, :, :, :, :, i6, :, i1, i2], axis=5)
            temp = np.mean(data[i0, :, 1:, :, :, 1:, :, :, i1, i2], axis=6)
            # print temp.shape
            if i2 == 1:
                vals[i0, i1, 0] = np.amin(temp)
                vals[i0, i1, 1] = np.amax(temp)
                temp1 = np.mean(data[i0, inds[0], :, inds[2], inds[3], :, :, :, i1, i2], axis=2)
                vals[i0, i1, 2] = np.amin(temp1)
                vals[i0, i1, 3] = np.amax(temp1)
            # print "Growth condition", i0, "Celltype", celltype[i1], "Sim type", sims[i2]
figs = []
cells = ['Mothers', 'Daughters']
lab = ['min 3xSEM', 'max 3xSEM', 'constrained min', 'constrained max']
NAMES_1 = list(['Dip Glu', 'Dip Gal', 'Dip Gly', 'Dip LGl', 'Dip Raf'])
fig = plt.figure(figsize=[16, 6])
plotrange = range(1, vals.shape[0])
for i0 in range(vals.shape[1]):
    ax = plt.subplot(1, 2, 1+i0)
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    marks = ['ro', 'bv', 'gD', 'ko']
    for i1 in range(vals.shape[2]):
        plt.plot(np.linspace(1.0, len(plotrange), len(plotrange)), vals[plotrange[0]:, i0, i1], marks[i1], ms=10, label=lab[i1])
    plt.legend()
    ax.set_title(cells[i0] + '. Unconstrained: $\sigma_K$, $\sigma_i$', size=16, weight="bold")
    ax.set_ylabel('$V_b$ $V_d$ slope ranges', size=16, weight="bold")
    plt.margins(0.2)
    plt.xticks(np.linspace(1.0, len(plotrange), len(plotrange)), [NAMES_1[i1] for i1 in plotrange], rotation='vertical')
#     fig.savefig('./plots_unshared/figure'+str(i1)+'.eps', bbox_inches='tight', dpi=fig.dpi)
# fig.savefig('./lab_meeting_figures/Feb17_test_models4_allmedia_fullvar.eps', bbox_inches='tight', dpi=fig.dpi)
fig.savefig('./lab_meeting_figures/Feb17_test_models4_nogluc_fullvar.eps', bbox_inches='tight', dpi=fig.dpi)
print "I made it"
# plt.show(fig)
