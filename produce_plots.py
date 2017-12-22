#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import matplotlib.pyplot as plt
import seaborn as sns


par1 = dict([('g1_std', 0.0), ('g2_std', 0.05), ('g1_thresh_std', 0.05), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0) \
                , ('initiator', 0), ('CD', 0.5), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 0)])

# Check the variation of whi5 with noise
obs_leafcells=np.load('obs_var_leafcells.npy')
par1['K']=(2 ** (1 - par1['CD'] / 1.0))/par1['CD']
g1_std = np.linspace(0.0, 0.15, 16)
g2_std = np.linspace(0.0, 0.15, 16)
fig = g.plot_systematically(obs_leafcells[::2,:,:,:],g1_std[::2],g2_std,par1)
fig.savefig('wb_varnoise_leafcells.eps', dpi=fig.dpi)

# Check the variation of whi5 with noise
obs_leafcells=np.load('obs_var_leafcells_discrgen.npy')
par1['K']=(2 ** (1 - par1['CD'] / 1.0))/par1['CD']
g1_std = np.linspace(0.0, 0.15, 16)
g2_std = np.linspace(0.0, 0.15, 16)
fig = g.plot_systematically(obs_leafcells[::2,:,:,:],g1_std[::2],g2_std,par1)
fig.savefig('wb_varnoise_discrgen_leafcells.eps', dpi=fig.dpi)

# Check the deviation from the model for whi5 average at birth
obs_leafcells=np.load('obs_var_leafcells_discrgen.npy')
par1['K']=(2 ** (1 - par1['CD'] / 1.0))/par1['CD']
g1_std = np.linspace(0.0, 0.15, 16)
g2_std = np.linspace(0.0, 0.15, 16)
fig = g.test_deviation(obs_leafcells,g2_std,par1)
fig.savefig('wb_deviation_model.eps', dpi=fig.dpi)

# Check the variation of whi5 with noise
obs_tree = np.load('obs_var_discrgen_wholepop.npy')
par1['K'] = 1.0
g1_std = np.linspace(0.0, 0.23, 24)
g2_std = np.linspace(0.0, 0.23, 24)
fig = g.plot_systematically(obs_tree, g1_std, g2_std, par1)
fig.savefig('wb_varnoise_tree_discrgen_changedinit.eps', dpi=fig.dpi)
figs = g.heat_maps_mother_daughter(obs_tree[::-2, ::2, :, :], g1_std[::2], g2_std[::2], 'Whi5 noiseless adder')
figs[0].savefig('Mother_vdvbslope_heatmap_discrgen.eps')
figs[1].savefig('Daughter_vdvbslope_heatmap_discrgen.eps')

fig = g.plot_vb_systematically(obs_tree, g1_std, g2_std, par1)
fig.savefig('vb_varnoise_tree_discrgen_changedinit.eps', dpi=fig.dpi)

vec = [0, 5, 10, 15, 20]  # selects the g1_std values which will be used
fig1, fig2 = g.test_function_syst(obs_tree, par1, g1_std, g2_std, vec)
fig1.savefig('syst_test_tree_discrgen_changedinit_daughters.eps', dpi=fig.dpi)
fig2.savefig('syst_test_tree_discrgen_changedinit_mothers.eps', dpi=fig.dpi)

# Check the variation of whi5 with variation in CD period
obs_discrgen = np.load('obs_var_discrgen_tree_cd0_5.npy')
par1['K'] = 1.0
par1['CD'] = 0.5
g1_std = np.linspace(0.0, 0.14, 8)
g2_std = np.linspace(0.0, 0.14, 8)
fig = g.plot_systematically(obs_discrgen, g1_std, g2_std, par1)
fig.savefig('tree_discrgen_CD_05.eps', dpi=fig.dpi)
vec = [0, 2, 4, 6]  # selects the g1_std values which will be used
fig1, fig2 = g.test_function_syst(obs_discrgen, par1, g1_std, g2_std, vec)
fig1.savefig('tree_discrgen_CD_05_testsyst_daughters.eps', dpi=fig1.dpi)
fig2.savefig('tree_discrgen_CD_05_testsyst_mothers.eps', dpi=fig2.dpi)

# Check the variation of whi5 for a noisy
obs_tree = np.load('whi5_noise_discrgen_tree_cd_75.npy')
par1['K'] = 1.0
g1_std = np.linspace(0.0, 0.23, 24)
g2_std = np.linspace(0.0, 0.23, 24)
figs = g.heat_maps_mother_daughter(obs_tree[::-2, ::2, :, :], g1_std[::2], g2_std[::2], 'Whi5 noisy adder')
figs[0].savefig('Mother_vdvbslope_heatmap_discrgen_noisyadder.eps')
figs[1].savefig('Daughter_vdvbslope_heatmap_discrgen_noisyadder.eps')