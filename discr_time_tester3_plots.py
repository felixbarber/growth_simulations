#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)

g1_std = np.linspace(0.0, 0.25, 6)
g2_std = np.linspace(0.0, 0.25, 6)
l_std = np.linspace(0.0, 0.25, 6)
cd = np.linspace(0.5, 1.0, 8)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.1), ('g1_thresh_std', 0.1), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 5), ('k_std', 0.0)])

# generating model predictions
a = 20
g1_std_m = np.linspace(0.0, 0.25, 6)
g2_std_m = np.linspace(0.0, 0.25, 6)
cd_m = np.linspace(0.5, 1.0, a)
l_m = l_std

# slopes = np.zeros([a, 5, 5, 2])
# for k in range(a):
#     par1['CD'] = cd_m[k]
#     for i in range(len(g1_std_m)):
#         par1['g1_thresh_std'] = g1_std_m[i]
#         #slopes[k, i, :, 0] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], g2_std_m)  # don't have the theory for gr yet
#         #slopes[k, i, :, 1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], g2_std_m)

# a = np.zeros((X, Y, Z, 6, 2, W))

labels = ['Discr time leaf', 'Discr time tree', 'Discr genr', 'Theory']
fullcelltype = ['Mothers', 'Daughters']
celltype = ['m', 'd']

slopes = np.zeros([len(cd_m), len(g1_std), len(g2_std_m), len(l_std), 2])
for k in range(len(cd_m)):
    par1['CD'] = cd_m[k]
    for i in range(len(g1_std_m)):
        par1['g1_thresh_std'] = g1_std_m[i]
        for j in range(len(l_std)):
            par1['l_std'] = l_std[j]
            slopes[k, i, :, j, 0] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], g2_std_m)
            slopes[k, i, :, j, 1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], g2_std_m)

obs = np.load('discr_time_tester1_model'+str(par1['modeltype'])+'_K_1.npy')
for k in range(2):
    fig = plt.figure(figsize=[20, 20])
    for i in range(len(g1_std)):
        for j in range(len(g2_std)):
            ax = fig.add_subplot(len(g1_std), len(g2_std), j+1+i*len(g2_std))
            ax.set_title('$\sigma_{i}/\Delta=$' + str(np.round(g1_std[i], 2)) + ' $\sigma_{G2}=$' +
                         str(np.round(g2_std[j], 2)))
            for h in range(len(l_std[::2])):
                ax.plot(cd, obs[:, i, j, 0, k, 2*h], linewidth=4.0, label=' Simulation '+'$\sigma_{\lambda}=$' +
                                                                          str(np.round(l_std[2*h], 2)))
                ax.plot(cd_m, slopes[:, i, j, 2*h, k], label=labels[3]+'$\sigma_{\lambda}=$' +
                        str(np.round(l_std[2*h], 2)))
            if i == len(g1_std)-1:
                ax.set_xlabel('Budded period $CD/t_d$')
            if j == 0:
                ax.set_ylabel('$V_d$ vs $V_b$ slope')
            ax.legend(loc=3)
    plt.suptitle('Noiseless adder model '+labels[2]+' Vd Vb slopes + noise in growth rate'+fullcelltype[k])
    fig.savefig('./discr_time_tester3_figs/vdvbslope_'+celltype[k]+'_model'+str(par1['modeltype'])+'.eps',
                bbox_inches='tight', dpi=fig.dpi)
# generating model predictions
a = 20
g1_std_m = g1_std
g2_std_m = np.linspace(0.0, 0.25, a)
cd_m = cd
l_m = l_std
slopes = np.zeros([len(cd_m), len(g1_std_m), len(g2_std_m), len(l_m), 2])
for k in range(len(cd_m)):
    par1['CD'] = cd_m[k]
    for i in range(len(g1_std_m)):
        par1['g1_thresh_std'] = g1_std_m[i]
        for j in range(len(l_m)):
            par1['l_std'] = l_m[j]
            slopes[k, i, :, j, 0] = g.slope_vbvd_m(par1, par1['g1_thresh_std'], g2_std_m)
            slopes[k, i, :, j, 1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], g2_std_m)
cd_ind = 4
for j in range(2):
    for k in range(2):
        fig = plt.figure(figsize=[6, 6])
        for i in range(len(g1_std)):
            plt.plot(g2_std, obs[cd_ind, i, :, 0, k, j], linewidth=4.0, label=' Simulation '+'$ \sigma_{i}/\Delta=$' +
                                                                              str(np.round(g1_std[i], 2)))
            plt.plot(g2_std_m, slopes[cd_ind, i, :, j, k], label=labels[3]+' $\sigma_{i}/\Delta=$' +
                                                                 str(np.round(g1_std_m[i], 2)))
        plt.title(fullcelltype[k] + ' $t_{G2}/t_d=$' + str(np.round(cd[cd_ind], 2)) + ', $\sigma_{\lambda}/\lambda=$'
                  + str(np.round(l_std[j], 2)))
        plt.legend(loc=4)
        plt.ylabel('$V_b$ $V_d$ regression slope')
        plt.xlabel('$\sigma_{G2}/t_d$')
        fig.savefig('./discr_time_tester3_figs/vdvbslope_'+celltype[k]+'_model'+str(par1['modeltype'])+'_lstd'+str(j) +
                    '_cd'+str(cd_ind)+'.pdf',   bbox_inches='tight', dpi=fig.dpi)

# def test_function_syst(obs, par1, g1_std, g2_std, vec):
#     # This function allows us to systematically simulate in order to check the
#     # agreement of different theoretical expressions with the simulations. We can see that this appears to be perfectly
#     # fine for the variation with respect to
#     labels = ['$V_d$ vs. $V_b$ slope', '$<V_b>$', '$<Vb^2>$', '$<V_d>$', '$<V_dV_b>$']
#     values = range(len(vec))
#     cmap = plt.get_cmap('cool')
#     cnorm = colors.Normalize(vmin=0, vmax=values[-1])
#     scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
#     fig1 = plt.figure(figsize=[15, 18])
#     for j in range(obs.shape[2] - 1):
#         plt.subplot(3, 2, j + 1)
#         for i in range(len(vec)):
#             colorval = scalarmap.to_rgba(values[i])
#             vals = np.empty((len(labels),len(g2_std)))
#             vals[1, :] = vb_func(par1,g1_std[vec[i]],g2_std)
#             vals[2, :] = vbvb_func(par1, g1_std[vec[i]], g2_std)
#             vals[3, :] = vd_func(par1, g1_std[vec[i]], g2_std)
#             vals[4, :] = vdvb_func(par1, g1_std[vec[i]], g2_std)
#             vals[0, :] = (vals[4, :]-vals[3, :] * vals[1, :])/(vals[2, :]-vals[1, :]**2)
#             plt.plot(g2_std, vals[j, :], label='theory $\sigma_{G1}=$ '+str(g1_std[vec[i]]))
#             plt.plot(g2_std, obs[vec[i], :, j, 1], color=colorval, label='sim $\sigma_{G1}=$ '+str(g1_std[vec[i]]))
#             plt.title(labels[j] + ' daughter ' + models[par1['modeltype']])
#             plt.xlabel('$\sigma_{G2}$')
#             plt.legend(loc=4)
#     fig2 = plt.figure(figsize=[15, 18])
#     if par1['mothervals']:
#         for j in range(obs.shape[2] - 1):
#             plt.subplot(3, 2, j + 1)
#             for i in range(len(vec)):
#                 colorval = scalarmap.to_rgba(values[i])
#                 vals = np.empty((len(labels), len(g2_std)))
#                 vals[1, :] = vb_m(par1, g1_std[vec[i]], g2_std)
#                 vals[2, :] = vbvb_m(par1, g1_std[vec[i]], g2_std)
#                 vals[3, :] = vd_m(par1, g1_std[vec[i]], g2_std)
#                 vals[4, :] = vdvb_m(par1, g1_std[vec[i]], g2_std)
#                 vals[0, :] = (vals[4, :]-vals[3, :] * vals[1, :])/(vals[2, :]-vals[1, :]**2)
#                 plt.plot(g2_std, vals[j, :], label='theory $\sigma_{G1}=$ ' + str(g1_std[vec[i]]))
#                 plt.plot(g2_std, obs[vec[i], :, j, 0], color=colorval, label='sim $\sigma_{G1}=$ ' + str(g1_std[vec[i]]))
#                 plt.title(labels[j] + ' mother '+models[par1['modeltype']])
#                 plt.xlabel('$\sigma_{G2}$')
#                 plt.legend(loc=4)
#     return fig1, fig2
