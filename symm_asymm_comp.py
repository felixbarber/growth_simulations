#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)

cd = np.linspace(0.5, 1.5, 41)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.1), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 0)])

slopes = np.empty([4, 2, len(cd)])
# slopes = np.load('symm_asymm_comp2.npy')
tg = [[[], [], []],[[], [], []]]
for k in range(len(cd)):
    par1['CD'] = cd[k]
    obs1, obs2, tg1, tg2 = g.single_par_meas5(par1)  # Obs1 are leaf cells, obs2 are entire tree cells
    slopes[0, :, k] = obs1[0, :]
    slopes[1, :, k] = obs2[0, :]
    obs3, tg3 = g.single_par_meas4(par1)
    tg[0][0].append(tg1[0])  # mothers data
    tg[0][1].append(tg2[0])
    tg[0][2].append(tg3[0])

    tg[1][0].append(tg1[1])  # daughters data
    tg[1][1].append(tg2[1])
    tg[1][2].append(tg3[1])

    slopes[2, :, k] = obs3[0, :]
    slopes[3, 0, k] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], par1['g2_std'])
    slopes[3, 1, k] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['g2_std'])
    del tg1, tg2, tg3, obs1, obs2, obs3
#
np.save('symm_asymm_comp4', slopes)
np.save('symm_asymm_tgrow_data', tg)
labels = ['Discretized time leaf cells', 'Discretized time tree cells', 'Discrete generations', 'Theory']
celltype = ['Mothers', 'Daughters']


fig = plt.figure(figsize=[16, 7])
for i in range(2):
    plt.subplot(1, 2, i+1)
    for j in range(slopes.shape[0]):
        plt.plot(cd, slopes[j, i, :], label=labels[j])
    plt.xlabel('Budded period $CD/t_d$')
    plt.ylabel('$V_d$ vs $V_b$ slope')
    plt.title(celltype[i])
    plt.legend()
plt.suptitle('Symmetric vs. Asymmetric division $\sigma_{CD}$ ='+str(par1['g1_thresh_std'])+' $\sigma_i$ ='
             + str(par1['g2_std']))
fig.savefig('symm_asymm_comp3.eps', bbox_inches='tight', dpi=fig.dpi)