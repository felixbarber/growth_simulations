#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

g1_std = np.linspace(0.0, 0.2, 9)
g2_std = np.linspace(0.0, 0.25, 6)
l_std = np.linspace(0.0, 0.25, 6)
cd = np.linspace(0.5, 1.0, 8)
f = np.linspace(0.25, 0.50, 11)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 800), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.685), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 11),
             ('l_std', 0.0), ('g1_delay', 0.0)])

models = [11, 12]
modeltype = ['min $V_s$', 'no min $V_s$']
# # GENERATING MODEL PREDICTIONS
# n = 1
# g1_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))
# g2_std_m = np.linspace(0.0, 0.25, 1+n*(len(g2_std)-1))
# cd_m = np.linspace(0.5, 1.0, 1+n*(len(cd)-1))
# l_std_m = np.linspace(0.0, 0.25, 1+n*(len(l_std)-1))
# k_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))

# LOADING SIMULATION DATA FROM MODEL5_SAMPLER.PY
obs = np.load('model11_12_K_1.npy')

# DIMENSIONS OF OBS MATRIX
# X, Y, Z, W, V = len(cd), len(g1_std), len(g2_std), len(l_std), len(f)
# a = np.zeros((X, Y, Z, W, V, len(models), 6, 2))

indices = [[3, 1, 1, 1], [3, 8, 1, 1], [3, 1, 5, 1], [3, 1, 1, 5], [3, 1, 5, 5]]

altmodels = [9, 4]
altmodeltype = ['Noisy div min $V_s$', 'Noisy div no min $V_s$']
slopes = []

values = range(2*len(indices)+len(altmodels)+1)  # 2x since 2 models being compared
cmap = plt.get_cmap('gnuplot2')
cnorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

# PLOTTTING
for k in range(len(celltype)):
    ind = 0
    ind1 = len(indices)
    fig = plt.figure(figsize=[8, 8])
    for i in range(len(indices)):
        slopes.append([])
        pars = indices[i]
        par1['CD'] = cd[pars[0]]
        par1['g1_thresh_std'] = g1_std[pars[1]]
        par1['g2_std'] = g2_std[pars[2]]
        par1['l_std'] = l_std[pars[3]]
        if k == 0:  # don't do it twice
            for j in range(len(altmodels)):
                slopes[i].append([])
                par1['modeltype'] = altmodels[j]
                obs_temp, tg_temp = g.single_par_meas4(par1)  # discretized gen
                slopes[i][j].append(obs_temp[0, :])  # mothers first, then daughters
        frac = np.array(1 - 2 ** (-par1['CD']))
        for j in range(len(models)):
            if j == 0:
                ind += 1
                index = ind
            if j == 1:
                ind1 += 1
                index = ind1
            colorval = scalarmap.to_rgba(values[index])
            plt.plot(f, obs[pars[0], pars[1], pars[2], pars[3], :, j, 0, k],
                     label=modeltype[j]+' $\sigma_i$='+str(np.round(par1['g1_thresh_std'], 2))
                     +' $\sigma_{G2}$=' + str(np.round(par1['g2_std'], 2))+' $\sigma_{\lambda}$='+
                           str(np.round(par1['l_std'], 2)), color=colorval)
            # plt.plot(frac, slopes[i][j][0][k], color=colorval, marker="o")
            par1['modeltype'] = 4  # print the model's predicted slope for each different condition
        print '$\sigma_i$=' + str(np.round(par1['g1_thresh_std'], 2)) + ' $\sigma_{G2}$=' + str(
            np.round(par1['g2_std'], 2)) + ' $\sigma_{\lambda}$=' + str(np.round(par1['l_std'], 2))
        if k==0:
            print celltype[k],g.slope_vbvd_func(par1, g1_std[pars[1]], g2_std[pars[2]])
        if k==1:
            print celltype[k], g.slope_vbvd_m(par1, g1_std[pars[1]], g2_std[pars[2]])
    plt.legend(loc=3)
    plt.title(fullcelltype[k]+' Constant adder model CD='+str(np.round(par1['CD'], 2)))
    plt.xlabel('Daughter volume fraction')
    plt.xlim(xmin=f[0]-0.02,xmax=f[-1]+0.02)
    plt.ylabel('$V_d$ vs. $V_b$ regression slope')
    fig.savefig('./modelfigs/model11_12_'+str(celltype[k])+'2.eps',   bbox_inches='tight', dpi=fig.dpi)
