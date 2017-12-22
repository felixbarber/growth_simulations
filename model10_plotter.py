#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)
labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

g1_std = np.linspace(0.0, 0.25, 6)
g2_std = np.linspace(0.0, 0.25, 6)
cd = np.linspace(0.5, 1.0, 8)
l_std = np.linspace(0.0, 0.2, 2)
k_std = np.linspace(0.0, 0.2, 2)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.1), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 10)
             , ('k_std', 0.0), ('l_std', 0.0), ('mothervals', 1)])

# GENERATING MODEL PREDICTIONS
n = 1
g1_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))
g2_std_m = np.linspace(0.0, 0.25, 1+n*(len(g2_std)-1))
cd_m = np.linspace(0.5, 1.0, 1+n*(len(cd)-1))
l_std_m = np.linspace(0.0, 0.2, 1+n*(len(l_std)-1))
k_std_m = np.linspace(0.0, 0.2, 1+n*(len(k_std)-1))

# LOADING SIMULATION DATA FROM MODEL10_SAMPLER.PY
obs = np.load('discr_time_tester1_model'+str(par1['modeltype'])+'_K_1.npy')
slopes = np.load('model'+str(par1['modeltype'])+'_theory_gen_numsamp'+str(n)+'.npy')


# PLOTTING.

# DIMENSIONS OF OBS MATRIX
# X, Y, Z, W, V = len(cd), len(g1_std), len(g2_std), len(l_std), len(k_std)
# a = np.zeros((X, Y, Z, 6, 2, W, V))
cd_ind = 3
k_std_ind = 0
l_std_ind = 0

for u in range(2):  # celltype
    fig = plt.figure(figsize=[6, 6])
    for i in range(len(g1_std)):
        # plt.plot(g2_std, obs[cd_ind, i, :, 0, u, l_std_ind, k_std_ind], linewidth=4.0,
        #          label=' Simulation '+'$ \sigma_{i}/\Delta=$' + str(np.round(g1_std[i], 2)))
        plt.plot(g2_std_m, slopes[i, :, n*cd_ind, n*l_std_ind, n*k_std_ind, u], linewidth=1.0,
                 label=labels[3]+' $\sigma_{i}/\Delta=$' + str(np.round(g1_std_m[i], 2)))
    plt.title(fullcelltype[u] + ' $t_{G2}/t_d=$' + str(np.round(cd[cd_ind], 2)) + ', $\sigma_{\lambda}/\lambda=$'
              + str(np.round(l_std[l_std_ind], 2)) + ', $sigma_{k}/K=$'+str(np.round(k_std[k_std_ind], 2)))
    plt.legend(loc=4)
    plt.ylabel('$V_b$ $V_d$ regression slope')
    plt.xlabel('$\sigma_{G2}/t_d$')
    fig.savefig('./modelfigs/model'+str(par1['modeltype'])+'vdvb_'+celltype[u]+'_lstd'
                +str(l_std_ind) + '_cd'+str(cd_ind)+'_kstd'+str(k_std_ind)+'.eps',   bbox_inches='tight', dpi=fig.dpi)

par1['CD'] = cd[cd_ind]
par1['k_std'] = k_std[k_std_ind]
par1['l_std'] = l_std[l_std_ind]
obs_new = np.empty(obs.shape[1:5])
obs_new[:, :, :, :] = obs[cd_ind, :, :, :, :, l_std_ind, k_std_ind]
# print obs.shape, obs_new.shape, par1['l_std'], par1['k_std'], par1['cd']

#par1['modeltype'] = 1

figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)-1))

u = 1
for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
    fig.savefig('./modelfigs/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                +str(l_std_ind) + '_cd'+str(cd_ind)+'_kstd'+str(k_std_ind)+'.eps',   bbox_inches='tight', dpi=fig.dpi)
    u += -1

# fig = plt.figure(figsize=[6,6])
# wbwb_5 = g.wbwb_p(par1, g1_std[1], g2_std_m)
# par1['modeltype'] = 1
# wbwb_1 = g.wbwb_p(par1, g1_std[1], g2_std_m)
# plt.plot(g2_std_m, wbwb_5, label='model 5')
# plt.plot(g2_std_m, wbwb_1, label='model 1')
# plt.legend()
# fig.savefig('./modelfigs/wbwb_comp.pdf',   bbox_inches='tight', dpi=fig.dpi)
