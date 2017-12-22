#!/usr/bin/env python

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g


def heat_map(obs, x, y, labels, celltype, vmax):
    # Assumes that mothers come first in the third dimension of obs
    fig = plt.figure(figsize=[16, 15])
    sns.heatmap(obs[::-1, :], xticklabels=np.around(x, decimals=2),
                yticklabels=np.around(y[::-1], decimals=2), annot=False, cmap='magma', vmin=0.0, vmax=vmax)
    plt.xlabel(labels[0], size=20)
    plt.ylabel(labels[1], size=20)
    plt.title(labels[2]+celltype, size=20)
    return fig

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 9),
             ('g1_delay', 0.0), ('l_std', 0.0)])

data = np.load('model_'+str(par1['modeltype'])+'_finegrid.npy')
low_conc_data = np.load('model_4_finegrid_lowconc.npy')

lab = ['$\sigma_{G2}/t_d$', '$t_{bud}/t_{d}$', '$V_d$ $V_b$ slope difference for constant adder w & w/o $V_s$ minimum']
lab1 = ['$\sigma_{G2}/t_d$', '$t_{bud}/t_d$', 'Mean total G1 volume added']
obs_labels =['Mean total G1 volume added / $\Delta$', '$\sigma/\Delta$ for total G1 volume added', 'CV of total G1 volume added']

# DATA SHAPE
cd = np.linspace(0.4, 1.0, 25)
g2_std = np.linspace(0.01, 0.25, 25)

celltype = ['m', 'd', 'p']
fullcelltype = [' mothers', ' daughters', ' population']

X0, X1 = len(cd), len(g2_std)
slopes = np.empty([X0, X1, len(celltype)])
g1_vol_obs = []
for i in range(3):
    g1_vol_obs.append(np.empty([X0, X1, len(celltype)]))
for i0 in range(X0):
    par1['CD'] = cd[i0]
    for i1 in range(X1):
        par1['g2_std'] = g2_std[i1]
        slopes[i0, i1, :] = g.slope_all_celltypes(par1, 0.0, g2_std[i1])
        for i2 in range(len(celltype)):
            g1_vol_obs[0][i0, i1, i2] = g.mean_vdvb(par1, cell_no=i2)/(par1['CD']*par1['K'])
            g1_vol_obs[1][i0, i1, i2] = g.std_vdvb(par1, cell_no=i2)/(par1['CD']*par1['K'])
            g1_vol_obs[2][i0, i1, i2] = g1_vol_obs[1][i0, i1, i2]/g1_vol_obs[0][i0, i1, i2]

for i0 in range(data.shape[-1]):
    a = np.zeros((X0, X1))
    a[:, :] = np.mean(data[:, :, :, 0, i0], 2)-slopes[:, :, i0]
    fig = heat_map(a, g2_std, cd, lab, fullcelltype[i0], vmax=1.0)
    fig.savefig('./modelfigs/model'+str(par1['modeltype'])+celltype[i0]+'.eps', bbox_inches='tight', dpi=fig.dpi)
    for i1 in range(len(g1_vol_obs)):
        a = np.zeros((X0, X1))
        a[:, :] = g1_vol_obs[i1][:, :, i0]
        lab1[2] = obs_labels[i1]
        fig1 = heat_map(a, g2_std, cd, lab1, fullcelltype[i0], vmax=np.amax(a))
        fig1.savefig('./modelfigs/model'+str(par1['modeltype'])+celltype[i0]+'_obs_'+str(i1)+'.eps', bbox_inches='tight'
                        , dpi=fig.dpi)
lab2 = lab[:2]
lab2.append('Percentage of cells with $[W_d]<1.0$')
for i0 in range(len(celltype)):
    par1['modeltype'] = 4
    a = np.zeros((X0, X1))
    a[:, :] = np.mean(low_conc_data[:, :, :, i0], 2)
    fig = heat_map(a, g2_std, cd, lab2, fullcelltype[i0], vmax=50.0)
    fig.savefig('./modelfigs/model' + str(par1['modeltype']) + celltype[i0] + 'lowconc.eps', bbox_inches='tight', dpi=fig.dpi)
