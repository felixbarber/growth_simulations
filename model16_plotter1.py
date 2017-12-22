#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx

font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)
labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

w_frac = np.linspace(0.325, 0.5, 8)
cd = np.linspace(0.45, 0.75, 2)
g1_std = np.linspace(0.0, 0.1, 3)
g2_std = np.linspace(0.0, 0.2, 6)
l_std = np.linspace(0.0, 0.2, 2)
k_std = np.linspace(0.0, 0.2, 2)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 10), ('l_std', 0.0),
             ('g1_delay', 0.0), ('k_std', 0.0), ('w_frac', 0.0), ('mothervals', True)])
models = [15, 16]

# LOADING SIMULATION DATA FROM MODEL5_SAMPLER.PY
obs = np.load('model_15_16_K_1.npy')

# PLOTTING.

# DIMENSIONS OF OBS MATRIX
X0, X1, X2, X3, X4, X5, X6 = len(w_frac), len(cd), len(g1_std), len(g2_std), len(l_std), len(k_std), len(models)
# a = np.zeros((X0, X1, X2, X3, X4, X5, X6, 6, 3))
f_ind = 6
cd_ind = 0
k_std_ind = 0
l_std_ind = 0
model_ind = 0

par1['w_frac'] = w_frac[f_ind]
par1['CD'] = cd[cd_ind]
par1['k_std'] = k_std[k_std_ind]
par1['l_std'] = l_std[l_std_ind]
par1['modeltype'] = models[model_ind]

obs_new = np.empty([X2, X3, 6, 2])
obs_new[:, :, :, :] = obs[f_ind, cd_ind, :, :, l_std_ind, k_std_ind, model_ind, :, :2]

figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))

u = 1
for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
    fig.savefig('./model15_figs/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                + str(l_std_ind) + '_cd'+str(cd_ind)+'_kstd'+str(k_std_ind)+'_f'+str(f_ind)+'.eps', bbox_inches='tight',
                dpi=fig.dpi)
    u += -1

it = [w_frac, cd, g1_std, g2_std, k_std, l_std]
list_it = ['w_frac', 'CD', 'g1_thresh_std', 'g2_std', 'k_std', 'l_std']
w_frac_m = np.linspace(w_frac[0], w_frac[-1], 40)
theory_slopes = np.empty([len(it), len(w_frac_m)])
ind = []  # stores the indices for the data to be plotted.

for i0 in range(len(it)):
    ind.append([])
    for i1 in range(len(list_it)):
        par1[list_it[i1]] = it[i1][0]  # reset everything every time
        ind[i0].append(0)
    if i0 > 0:
        par1[list_it[i0]] = it[i0][-1]  # first entry gives the zero vector
        ind[i0][i0] = -1
    for i1 in range(len(w_frac_m)):
        par1['w_frac'] = w_frac_m[i1]
        theory_slopes[i0, i1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['g2_std'])
values = range(len(it)+1)  # 2x since 2 models being compared
cmap = plt.get_cmap('gnuplot2')
cnorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

fig = plt.figure(figsize=[8, 8])
for i0 in range(len(it)):
    colorval = scalarmap.to_rgba(values[i0])
    plt.scatter(w_frac, obs[:, ind[i0][1], ind[i0][2], ind[i0][3], ind[i0][4], ind[i0][5], 1, 0, 1],
             label='sims'+' cd '+str(it[1][ind[i0][1]])+' g1 '+str(it[2][ind[i0][2]])+' g2 '
                   + str(it[3][ind[i0][3]])+' k '+str(it[4][ind[i0][4]])+' l '+str(it[5][ind[i0][5]]), color=colorval)
    plt.plot(w_frac_m, theory_slopes[i0, :], color=colorval)
plt.title('Daughter Noisy adder model with const Whi5 fraction f')
plt.xlabel('Daughter Whi5 fraction')
plt.ylabel('$V_d$ vs. $V_b$ regression slope')
plt.legend(loc=3)
fig.savefig('./model15_figs/model16_slopes.eps', bbox_inches='tight', dpi=fig.dpi)
