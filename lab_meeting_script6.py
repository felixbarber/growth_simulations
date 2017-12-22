#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx

par1 = dict([('g1_std', 0.0), ('g2_std', 0.15), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 100.0/0.67), ('td', 1.0), ('modeltype', 16), ('l_std', 0.2),
             ('g1_delay', 0.0), ('k_std', 0.0), ('w_frac', 0.0)])

models = [15, 16]
w_frac = np.linspace(0.2, 0.5, 13)
data = np.load('./lab_meeting_figures/model_16_numneg.npy')
X0, X1 = len(w_frac), len(models)
# slopes = np.zeros((X0, X1, 3))

values = range(4)  # 2x since 2 models being compared
cmap = plt.get_cmap('gnuplot2')
cnorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

celltype = ['Mothers', 'Daughters', 'Population']


for i0 in range(2):
    fig = plt.figure(figsize=[8, 8])
    for i1 in range(3):
        colorval = scalarmap.to_rgba(values[i1])
        plt.scatter(w_frac, data[:, i0, i1],
                 label=celltype[i1]+' $t_b$= '+str(0.67)+', $\sigma_i=$ '+str(0.0)+', $\sigma_{G2}/t_d=$ '
                       + str(0.15)+', $\sigma_k/k=$ '+str(0.0)+', $\sigma_{\lambda}/\lambda=$ '+str(0.2), color=colorval)
        plt.plot(((2**par1['CD']-1)/2**par1['CD'], (2**par1['CD']-1)/2**par1['CD']), (0.0, 5.0), 'k-')
    plt.title('Noisy synthesis rate model '+str(models[i0]))
    plt.xlabel('Daughter Whi5 fraction')
    plt.ylabel('Percentage of low concentration cells')
    plt.legend(loc=1)
    fig.savefig('./lab_meeting_figures/model'+str(models[i0])+'lowconc.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig
