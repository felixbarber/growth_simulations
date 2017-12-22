#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g


model1_data = np.load('whi5_noisy_adder_discrgen_tree_cd_75_K_1.npy')
g1_std = np.linspace(0.0, 0.62, 32)
g2_std = np.linspace(0.0, 0.62, 32)

if len(g1_std) != model1_data.shape[0] or len(g2_std) != model1_data.shape[1]:
    raise ValueError('Check your simulation data')

par1 = dict([('g1_std', 0.0), ('g2_std', 0.05), ('g1_thresh_std', 0.05), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0) \
                , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 1), ('mothervals', 0)])

par1['modeltype'] = 1  # keeps track of the model number we are looking at. Number zero is for a noiseless Whi5 adder,
    #  while 1 is for a noisy Whi5 adder

vec = [0, 7, 15, 23, 31]  # selects the g1_std values which will be used
fig1, fig2 = g.test_function_syst(model1_data, par1, g1_std, g2_std, vec)
fig1.savefig('tree_discrgen_model1_CD_075_k_1_daughters.eps', dpi=fig1.dpi)
fig2.savefig('tree_discrgen_model1_CD_075_k_1_mothers.eps', dpi=fig2.dpi)
