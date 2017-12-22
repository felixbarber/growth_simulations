#!/usr/bin/env python

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g


def heat_map(obs, x, y, labels, celltype, vmax, model_name):
    # Assumes that mothers come first in the third dimension of obs
    fig = plt.figure(figsize=[16, 15])
    sns.heatmap(obs[::-1, :], xticklabels=np.around(x, decimals=2),
                yticklabels=np.around(y[::-1], decimals=2), annot=False, cmap='magma', vmin=0.0, vmax=vmax)
    plt.xlabel(labels[0], size=20)
    plt.ylabel(labels[1], size=20)
    plt.title(labels[2]+model_name+celltype, size=20)
    return fig

par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('CD', 0.67), ('num_gen', 9), ('td', 1.0), ('modeltype', 4)
             , ('l_std', 0.0), ('g1_delay', 0.0), ('d_std', 0.2), ('K', 100.0/0.67), ('delta', 100.0), ('k_std', 0.0)])

cd = np.linspace(0.4, 1.0, 16)
g2_std = np.linspace(0.01, 0.25, 16)
models = [17, 18, 4, 9, 5, 10]
num_reps = 5
X0, X1, X2, X3 = len(cd), len(g2_std), len(models), num_reps
vals = ['CD', 'g2_std', 'modeltype']
pars = [cd, g2_std, models]
# a = np.zeros((X0, X1, X2, X3, 2, 3))
slopes = np.zeros((X0, X1, X2, 2))  # this value will store the theoretically predicted slopes
# data = np.load('./lab_meeting_figures/modelcomp_finegrid.npy')
data = np.load('./lab_meeting_figures/modelcomp_finegrid_g1_0.npy')
print data.shape

lab = ['$\sigma_{G2}/t_d$', '$t_{bud}/t_{d}$', '$V_d$ $V_b$ slope difference w & w/o $V_s$ minimum, model']
lab1 = ['$\sigma_{G2}/t_d$', '$t_{bud}/t_d$', 'Mean total G1 volume added']
obs_labels =['Mean total G1 volume added / $\Delta$', '$\sigma/\Delta$ for total G1 volume added', 'CV of total G1 volume added']

celltype = ['m', 'd', 'p']
fullcelltype = [' mothers', ' daughters', ' population']

# g1_vol_obs = []
# for i in range(3):
#     g1_vol_obs.append(np.empty([X0, X1, len(celltype)]))
for i0 in range(X0):
    par1[vals[0]] = pars[0][i0]
    for i1 in range(X1):
        par1[vals[1]] = pars[1][i1]
        for i2 in range(X2):
            par1[vals[2]] = pars[2][i2]
            slopes[i0, i1, i2, 0] = g.slope_vbvd_m(par1, 0.0, par1['td'] * g2_std[i1])
            slopes[i0, i1, i2, 1] = g.slope_vbvd_func(par1, 0.0, par1['td']*g2_std[i1])

temp = 1
for i0 in range(len(models)):
    for temp in range(len(celltype)-1):
        a = np.zeros((X0, X1))
        # print data[:, :, i0, :, 1, temp].shape
        a[:, :] = np.mean(data[:, :, i0, :, 1, temp], 2)-slopes[:, :, i0, temp]
        fig = heat_map(a, g2_std, cd, lab, fullcelltype[temp], vmax=1.0, model_name=str(models[i0]))
        fig.savefig('./lab_meeting_figures/slopediff_model'+str(models[i0])+celltype[temp]+'.eps', bbox_inches='tight', dpi=fig.dpi)
        del fig
        # for i1 in range(len(g1_vol_obs)):
        #     a = np.zeros((X0, X1))
        #     a[:, :] = g1_vol_obs[i1][:, :, i0]
        #     lab1[2] = obs_labels[i1]
        #     fig1 = heat_map(a, g2_std, cd, lab1, fullcelltype[i0], vmax=np.amax(a))
        #     fig1.savefig('./modelfigs/model'+str(par1['modeltype'])+celltype[i0]+'_obs_'+str(i1)+'.eps', bbox_inches='tight'
        #                     , dpi=fig.dpi)
lab2 = lab[:2]
lab2.append('Percentage of cells with $[W_d]<1.0$, model')
for i0 in range(len(models)):
    for i1 in range(len(celltype)):
        a = np.zeros((X0, X1))
        a[:, :] = np.mean(data[:, :, i0, :, 0, i1], 2)
        fig = heat_map(a, g2_std, cd, lab2, fullcelltype[i1], vmax=50.0, model_name=str(models[i0]))
        fig.savefig('./lab_meeting_figures/lowconc_G10_model' + str(models[i0]) + celltype[i1] + '.eps',
                    bbox_inches='tight', dpi=fig.dpi)
        del fig
