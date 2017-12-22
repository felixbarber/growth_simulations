#!/usr/bin/env python

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g


def heat_map(obs, x, y, labels, celltype):
    # Assumes that mothers come first in the third dimension of obs
    fig = plt.figure(figsize=[16, 15])
    sns.heatmap(obs[::-1, :], xticklabels=np.around(x, decimals=2),
                yticklabels=np.around(y[::-1], decimals=2), annot=False, cmap='magma', vmin=0.0, vmax=1.0)
    plt.xlabel(labels[0], size=20)
    plt.ylabel(labels[1], size=20)
    plt.title(labels[2]+celltype, size=20)
    return fig

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 13),
             ('g1_delay', 0.0), ('l_std', 0.0), ('w_frac', 0.4)])

lab1 = ['$\sigma_{G2}/t_d$', '$t_{bud}/t_d$', 'Mean total G1 volume added']
obs_labels =['Mean total G1 volume added / $\Delta$', '$\sigma/\Delta$ for total G1 volume added', 'CV of total G1 volume added']

# DATA SHAPE
cd = np.linspace(0.4, 1.0, 25)
g2_std = np.linspace(0.01, 0.25, 25)
f = np.linspace(0.3, 0.5, 9)

celltype = ['m', 'd', 'p']
fullcelltype = [' mothers', ' daughters', ' population']

X0, X1, X2 = len(cd), len(g2_std), len(f)
g1_vol_obs = []
for i in range(3):
    g1_vol_obs.append(np.empty([X0, X1, X2, len(celltype)]))
for i0 in range(X0):
    par1['CD'] = cd[i0]
    for i1 in range(X1):
        par1['g2_std'] = g2_std[i1]
        for i2 in range(X2):
            par1['w_frac'] = f[i2]
            for i3 in range(len(celltype)):
                g1_vol_obs[0][i0, i1, i2, i3] = g.mean_vdvb(par1, cell_no=i3)/(par1['CD']*par1['K'])
                g1_vol_obs[1][i0, i1, i2, i3] = g.std_vdvb(par1, cell_no=i3)/(par1['CD']*par1['K'])
                g1_vol_obs[2][i0, i1, i2, i3] = g1_vol_obs[1][i0, i1, i2, i3]/g1_vol_obs[0][i0, i1, i2, i3]

for i0 in range(3):
    for i1 in range(len(g1_vol_obs)):
        for i2 in range(X2):
            a = np.zeros((X0, X1))
            a[:, :] = g1_vol_obs[i1][:, :, i2, i0]
            lab1[2] = obs_labels[i1]
            fig1 = heat_map(a, g2_std, cd, lab1, fullcelltype[i0])
            fig1.savefig('./model13figs/model'+str(par1['modeltype'])+celltype[i0]+'_obs_'+str(i1)+'_f_'+str(i2)+'.eps', bbox_inches='tight'
                            , dpi=fig1.dpi)
