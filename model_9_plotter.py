#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np

labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

g1_std = np.linspace(0.0, 0.25, 6)
g2_std = np.linspace(0.0, 0.25, 6)
l_std = np.linspace(0.0, 0.25, 6)
cd = np.linspace(0.5, 1.0, 8)
k_vals = np.linspace(0.5, 1.5, 3)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 12), ('K', 1.0), ('td', 1.0), ('modeltype', 4), ('l_std', 0.0),
             ('mothervals', True)])

# GENERATING MODEL PREDICTIONS
n = 1
g1_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))
g2_std_m = np.linspace(0.0, 0.25, 1+n*(len(g2_std)-1))
cd_m = np.linspace(0.5, 1.0, 1+n*(len(cd)-1))
l_std_m = np.linspace(0.0, 0.25, 1+n*(len(l_std)-1))
k_std_m = np.linspace(0.0, 0.25, 1+n*(len(g1_std)-1))

# LOADING SIMULATION DATA FROM MODEL5_SAMPLER.PY
obs = np.load('discr_time_tester3_model'+str(par1['modeltype'])+'.npy')

# DIMENSIONS OF OBS MATRIX
# X, Y, Z, W, V = len(cd), len(g1_std), len(g2_std), len(l_std), len(k_vals)
# a = np.zeros((X, Y, Z, 6, 2, W, V))

cd_ind = 3
k_ind = 1
l_std_ind = 0

par1['CD'] = cd[cd_ind]
par1['K'] = k_vals[k_ind]
par1['l_std'] = l_std[l_std_ind]
obs_new = np.empty(obs.shape[1:5])
obs_new[:, :, :, :] = obs[cd_ind, :, :, :, :, l_std_ind, k_ind]
# print obs.shape, obs_new.shape, par1['l_std'], par1['k_std'], par1['cd']

#par1['modeltype'] = 1

figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)-1))

u = 1
for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
    fig.savefig('./modelfigs/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                +str(l_std_ind) + '_cd'+str(cd_ind)+'_k'+str(k_ind)+'.eps',   bbox_inches='tight', dpi=fig.dpi)
    u += -1
