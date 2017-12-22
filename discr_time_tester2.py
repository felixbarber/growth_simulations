#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
import seaborn as sns
import os.path

# These values should be the same as those in discr_time_tester1 (or whatever script produced these files)
g1_std = np.linspace(0.0, 0.28, 15)
g2_std = np.linspace(0.0, 0.28, 15)
cd = np.linspace(0.5, 1.5, 16)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 0)])


celltype = ['m', 'd']
fullcelltype = ['Mothers', 'Daughters']
cdvec = [0, 1, 2, 3, 4, 5, 6, 7]
g1vec = [0, 2, 4, 6]
g2vec = [0, 2, 4, 6]

for i in cdvec:
    for l in range(2):
        fig = plt.figure(figsize=[16, 16])
        for j in range(len(g1vec)):
            for k in range(len(g2vec)):
                ax = fig.add_subplot(len(g1vec), len(g2vec), k + 1 + j * len(g2vec))
                basepath = '~/simulation_data/discr_time_tester1_data/'
                path = basepath + 'tgrow_'+celltype[l]+'_model0_discr_time_leaf_cd_' + str(i) \
                     + '_s1_' + str(g1vec[j]) + '_s2_' + str(g2vec[k]) + '.npy'
                if os.path.isfile(path):
                    tg = np.load(path)
                    tg = tg[~np.isnan(tg)]
                    ax = sns.distplot(tg, label='discr time leaf', ax=ax)
                    del tg
                path = basepath + 'tgrow_'+celltype[l]+'_model0_discr_time_tree_cd_' + str(i) \
                       + '_s1_' + str(g1vec[j]) + '_s2_' + str(g2vec[k]) + '.npy'
                if os.path.isfile(path):
                    tg = np.load(path)
                    tg = tg[~np.isnan(tg)]
                    ax = sns.distplot(tg, label='discr time tree', ax=ax)
                    del tg
                path = basepath + 'tgrow_'+celltype[l]+'_model0_discr_genr_tree_cd_' + str(i) \
                       + '_s1_' + str(g1vec[j]) + '_s2_' + str(g2vec[k]) + '.npy'
                if os.path.isfile(path):
                    tg = np.load(path)
                    tg = tg[~np.isnan(tg)]
                    ax = sns.distplot(tg, label='discr genr tree', ax=ax)
                    del tg
                ax.set_title('$\sigma_{i}=$' + str(np.round(g1_std[g1vec[j]], 2)) + ' $\sigma_{G2}=$' +
                          str(np.round(g2_std[g2vec[k]], 2)))
                if j == len(g1vec) - 1:
                    plt.xlabel('Growth time $t/t_d$')
                plt.legend()
            print "Number of rows left =" + str(len(g1vec)-j-1)
        plt.suptitle(fullcelltype[l]+' $t_{g}$ dists $CD/t_d=$'+str(np.round(cd[i], 2))+' $K=$'+str(par1['K']))
        fig.savefig('./discr_time_tester2_figs/tgrow_'+celltype[l]+'_model0_cd_' + str(i) + '.eps', dpi=fig.dpi)
        print str(l+1) + " figures done"
