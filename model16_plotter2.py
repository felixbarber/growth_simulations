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

cd = np.linspace(0.4, 0.75, 8)
g1_std = np.linspace(0.0, 0.1, 5)
g2_std = np.linspace(0.0, 0.2, 5)
l_std = np.linspace(0.0, 0.2, 5)
k_std = np.linspace(0.0, 0.2, 3)
w_frac = np.linspace(0.2, 0.5, 13)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 16), ('l_std', 0.0),
             ('g1_delay', 0.0), ('k_std', 0.0), ('w_frac', 0.0), ('mothervals', True)])
models = [15, 16]

# LOADING SIMULATION DATA FROM MODEL5_SAMPLER.PY
obs = np.load('model_16_K_1_sampler.npy')

# PLOTTING.

# DIMENSIONS OF OBS MATRIX
X0, X1, X2, X3, X4, X5 = len(cd), len(g1_std), len(g2_std), len(l_std), len(k_std), len(w_frac)
# a = np.zeros((X0, X1, X2, X3, X4, X5, 6, 3))
f_ind = 11
cd_ind = 5
k_std_ind = 0
l_std_ind = 0

par1['w_frac'] = w_frac[f_ind]
par1['CD'] = cd[cd_ind]
par1['k_std'] = k_std[k_std_ind]
par1['l_std'] = l_std[l_std_ind]

obs_new = np.empty([X1, X2, 6, 2])
obs_new[:, :, :, :] = obs[cd_ind, :, :, l_std_ind, k_std_ind, f_ind, :, :2]

figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))

u = 1
for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
    fig.savefig('./model15_figs/model'+str(par1['modeltype'])+'V2syst_test_'+celltype[u]+'_lstd'
                + str(l_std_ind) + '_cd'+str(cd_ind)+'_kstd'+str(k_std_ind)+'_f'+str(f_ind)+'.eps', bbox_inches='tight',
                dpi=fig.dpi)
    u += -1

it = [cd, g1_std, g2_std, l_std, k_std]
list_it = ['CD', 'g1_thresh_std', 'g2_std', 'l_std', 'k_std']
w_ind = 5
w_frac_m = np.linspace(w_frac[w_ind], w_frac[-1], 40)
ind = [[5, 0, 3, 0, 0], [5, 1, 3, 0, 0], [5, 0, 3, 1, 0], [5, 0, 3, 0, 1]]  # stores the indices for the data to be plotted.
theory_slopes = np.empty([len(ind), len(w_frac_m)])


for i0 in range(len(ind)):
    for i1 in range(len(list_it)):
        par1[list_it[i1]] = it[i1][ind[i0][i1]]
    for i1 in range(len(w_frac_m)):
        par1['w_frac'] = w_frac_m[i1]
        theory_slopes[i0, i1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['g2_std'])
values = range(len(it)+1)  # 2x since 2 models being compared
cmap = plt.get_cmap('gnuplot2')
cnorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

fig = plt.figure(figsize=[8, 8])
for i0 in range(len(ind)):
    colorval = scalarmap.to_rgba(values[i0])
    plt.scatter(w_frac[w_ind:], obs[ind[i0][0], ind[i0][1], ind[i0][2], ind[i0][3], ind[i0][4], w_ind:, 0, 1],
             label='sims'+' $t_b$= '+str(it[0][ind[i0][0]])+', $\sigma_i=$ '+str(it[1][ind[i0][1]])+', $\sigma_{G2}/t_d=$ '
                   + str(it[2][ind[i0][2]])+', $\sigma_k/k=$ '+str(it[3][ind[i0][3]])+', $\sigma_{\lambda}/\lambda=$ '+str(it[4][ind[i0][4]]), color=colorval)
    plt.plot(w_frac_m, theory_slopes[i0, :], color=colorval)
plt.plot(((2**cd[ind[0][0]]-1)/2**cd[ind[0][0]], (2**cd[ind[0][0]]-1)/2**cd[ind[0][0]]), (0.95, 1.05), 'k-')
plt.title('Daughter Noisy adder model with const Whi5 fraction f')
plt.xlabel('Daughter Whi5 fraction')
plt.ylabel('$V_d$ vs. $V_b$ regression slope')
plt.legend(loc=1)
fig.savefig('./model15_figs/model16_slopes_1.eps', bbox_inches='tight', dpi=fig.dpi)

# vals = np.empty([len(ind),len(w_frac[w_ind:])])
# par1['modeltype'] = 14
# for i0 in range(len(ind)):
#     for i1 in range(len(list_it)):
#         par1[list_it[i1]] = it[i1][ind[i0][i1]]
#     for i1 in range(len(w_frac[w_ind:])):
#         par1['w_frac'] = w_frac[w_ind + i1]
#         obs = g.single_par_meas6(par1)
#         vals[i0,i1] = obs[0, 1]
# np.save('./model15_figs/model14_slopes_1', vals)

vals=np.load('.//model15_figs/model14_slopes_1.npy')
fig = plt.figure(figsize=[8, 8])
for i0 in range(len(ind)):
    colorval = scalarmap.to_rgba(values[i0])
    plt.scatter(w_frac[w_ind:], vals[i0,:],
             label='sims'+' $t_b$= '+str(it[0][ind[i0][0]])+', $\sigma_i=$ '+str(it[1][ind[i0][1]])+', $\sigma_{G2}/t_d=$ '
                   + str(it[2][ind[i0][2]])+', $\sigma_k/k=$ '+str(it[3][ind[i0][3]])+', $\sigma_{\lambda}/\lambda=$ '+str(it[4][ind[i0][4]]), color=colorval)
plt.plot(((2**cd[ind[0][0]]-1)/2**cd[ind[0][0]], (2**cd[ind[0][0]]-1)/2**cd[ind[0][0]]), (0.95, 1.05), 'k-')
plt.title('Daughter Constant adder model with const Whi5 fraction f')
plt.xlabel('Daughter Whi5 fraction')
plt.ylabel('$V_d$ vs. $V_b$ regression slope')
plt.legend(loc=1)
fig.savefig('./model15_figs/model14_slopes_1.eps', bbox_inches='tight', dpi=fig.dpi)

