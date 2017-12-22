#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import scipy
plt.rc('text', usetex=True)

def gen_fig(var):
    labels = [r'$\sigma_i/\langle A_c\rangle$', '$\sigma_{t}/t_{db}$']
    titles = ['$f=0.5$, $(C+D)/t_{db}=0.7$', '$f=0.5$, $(C+D)/t_{db}=0.9$']
    tit_num = ['(A)', '(B)', '(C)', '(D)']
    cmap_lims = [0.0, 2.0]
    ticks = np.linspace(0.01, 0.3, num=30)
    figs=[]
    for i0 in range(len(var)):
        print var[i0]
        temp = np.load(var[i0] + '.npy')

        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1, 1, 1)
        a = np.absolute(temp - 1.0) < 0.1
        ax = plt.imshow(scipy.ndimage.morphology.binary_fill_holes(a))
        fig.savefig(var[i0]+'_bin.png', bbox_inches='tight', dpi=fig.dpi)
        del fig
        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1,1,1)
        ax = g.heat_map_pd(temp, ticks, ticks, ax, xlabel=labels[0], ylabel=labels[1], title=tit_num[i0] + titles[np.mod(i0, 2)],
                          cmap_lims=cmap_lims)
        fig.savefig(var[i0]+'.eps', bbox_inches='tight', dpi=fig.dpi)
        del temp, fig
    return var

def gen_fig_1(var):
    labels = [r'$\sigma_s/\langle \tilde{\Delta}\rangle$', '$\sigma_{t}/t_{db}$']
    titles = ['$f=0.5$, $(C+D)/t_{db}=0.7$', '$f=0.5$, $(C+D)/t_{db}=0.9$']
    tit_num = ['(A)', '(B)', '(C)', '(D)']
    cmap_lims = [0.0, 2.0]
    ticks = np.linspace(0.01, 0.3, num=30)
    figs=[]
    for i0 in range(len(var)):
        print var[i0]
        temp = np.load(var[i0] + '.npy')

        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1, 1, 1)
        a = np.absolute(temp - 1.0) < 0.1
        ax = plt.imshow(scipy.ndimage.morphology.binary_fill_holes(a))
        fig.savefig(var[i0]+'_bin.png', bbox_inches='tight', dpi=fig.dpi)
        del fig
        fig = plt.figure(figsize=[8, 8])
        ax = plt.subplot(1,1,1)
        ax = g.heat_map_pd(temp, ticks, ticks, ax, xlabel=labels[0], ylabel=labels[1], title=tit_num[i0] + titles[np.mod(i0, 2)],
                          cmap_lims=cmap_lims)
	plt.show()
        fig.savefig(var[i0]+'.eps', bbox_inches='tight', dpi=fig.dpi)
        del temp, fig
    return var

figs = []

# Figure 4 actually fig number 3

#paths = ['./po_yi_data/Fig3_1_1', './po_yi_data/Fig3_1_2', './po_yi_data/Fig3_2_1', './po_yi_data/Fig3_2_2']
#save_path = './po_yi_plots/fig3_v1'
#figs.append(gen_fig(paths))

# Figure 8 actually fig number 6

#paths = ['./po_yi_data/Fig5_1', './po_yi_data/Fig5_2']
#save_path = './po_yi_plots/fig6_v1'
#figs.append(gen_fig(paths))

paths = ['./po_yi_data_final/Fig5_1_1', './po_yi_data_final/Fig5_1_2','./po_yi_data_final/Fig5_2_1', './po_yi_data_final/Fig5_2_2']
save_path = './po_yi_data_final/fig5'
figs.append(gen_fig_1(paths))

paths = ['./po_yi_data_final/Fig6_1', './po_yi_data_final/Fig6_2']
save_path = './po_yi_data_final/fig6'
figs.append(gen_fig(paths))

