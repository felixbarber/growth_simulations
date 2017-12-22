#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)

g1_std = np.linspace(0.05, 0.25, 5)
g2_std = np.linspace(0.05, 0.25, 5)
cd = np.linspace(0.5, 1.0, 8)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.1), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 0)])

# generating model predictions
a = 20
g1_std_m = np.linspace(0.05, 0.25, 5)
g2_std_m = np.linspace(0.05, 0.25, 5)
cd_m = np.linspace(0.5, 1.0, a)
obs = np.load('discr_time_tester1_model'+str(par1['modeltype'])+'_K_1.npy')
slopes = np.zeros([a, 5, 5, 2])
for k in range(a):
    par1['CD'] = cd_m[k]
    for i in range(len(g1_std_m)):
        par1['g1_thresh_std'] = g1_std_m[i]
        slopes[k, i, :, 0] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], g2_std_m)
        slopes[k, i, :, 1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], g2_std_m)

labels = ['Discr time leaf', 'Discr time tree', 'Discr genr tree', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

for k in range(2):
    fig = plt.figure(figsize=[16, 16])
    for i in range(len(g1_std)):
        for j in range(len(g2_std)):
            ax = fig.add_subplot(len(g1_std), len(g2_std), j+1+i*len(g2_std))
            ax.set_title('$\sigma_{i}=$' + str(np.round(g1_std[i], 2)) + ' $\sigma_{G2}=$' +
                         str(np.round(g2_std[j], 2)))
            #ax.plot(cd, obs[:, i, j, 0, k, 0], label=labels[0])
            ax.plot(cd, obs[:, i, j, 0, k, 1], linewidth=4.0, label=labels[1])
            ax.plot(cd, obs[:, i, j, 0, k, 2], linewidth=4.0, label=labels[2])
            ax.plot(cd_m, slopes[:, i, j, k], label=labels[3])
            if i == len(g1_std)-1:
                ax.set_xlabel('Budded period $CD/t_d$')
            if j == 0:
                ax.set_ylabel('$V_d$ vs $V_b$ slope')
            ax.legend(loc=3)
    plt.suptitle('Noiseless adder model Vd Vb slopes '+fullcelltype[k])
    fig.savefig('./discr_time_tester2_figs/vdvbslope_'+celltype[k]+'_model'+str(par1['modeltype'])+'.eps',
                bbox_inches='tight', dpi=fig.dpi)

# This part allows you to plot the distributions of growth times.

# n = 0
# for i in range(len(cd)):
#     for l in range(2):
#         n += 1
#         fig = plt.figure(figsize=[16, 16])
#         for j in range(len(g1_std)):
#             for k in range(len(g2_std)):
#                 ax = fig.add_subplot(len(g1_std), len(g2_std), k + 1 + j * len(g2_std))
#                 basepath = '/home/felix/simulation_data/discr_time_tester1_data/'
#                 path = basepath + 'tgrow_'+celltype[l]+'_model'+str(par1['modeltype'])+'_discr_time_leaf_cd_' + str(i) \
#                      + '_s1_' + str(j) + '_s2_' + str(k) + '.npy'
#                 if os.path.isfile(path):
#                     tg = np.load(path)
#                     tg = tg[~np.isnan(tg)]
#                     ax = sns.distplot(tg, label='discr time leaf', ax=ax)
#                     del tg
#                 path = basepath + 'tgrow_'+celltype[l]+'_model'+str(par1['modeltype'])+'_discr_time_tree_cd_' + str(i) \
#                        + '_s1_' + str(j) + '_s2_' + str(k) + '.npy'
#                 if os.path.isfile(path):
#                     tg = np.load(path)
#                     if len(tg) < 100:
#                         raise ValueError('Something be wrong')
#                     tg = tg[~np.isnan(tg)]
#                     ax = sns.distplot(tg, label='discr time tree', ax=ax)
#                     del tg
#                 path = basepath + 'tgrow_'+celltype[l]+'_model'+str(par1['modeltype'])+'_discr_genr_tree_cd_' + str(i) \
#                        + '_s1_' + str(j) + '_s2_' + str(k) + '.npy'
#                 if os.path.isfile(path):
#                     tg = np.load(path)
#                     tg = tg[~np.isnan(tg)]
#                     ax = sns.distplot(tg, label='discr genr tree', ax=ax)
#                     del tg
#                 ax.set_title('$\sigma_{i}=$' + str(np.round(g1_std[j], 2)) + ' $\sigma_{G2}=$' +
#                           str(np.round(g1_std[k], 2)))
#                 plt.xlim((-3, 5))
#                 if j == len(g1_std) - 1:
#                     plt.xlabel('Growth time $t/t_d$')
#                 plt.legend()
#             print "Number of rows left =" + str(len(g1_std)-j-1)
#         plt.suptitle(fullcelltype[l]+' $t_{g}$ dists $CD/t_d=$'+str(np.round(cd[i], 2))+' $K=$'+str(par1['K']))
#         fig.savefig('./discr_time_tester2_figs/tgrow_'+celltype[l]+'_model'+str(par1['modeltype'])+
#                     '_cd_' + str(i) + '.eps', dpi=fig.dpi)
#         print str(n) + " figures done"
