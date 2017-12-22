#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns

models_names = ['discr gen noiseless whi5 adder 0', 'discr gen noisy whi5 adder 1', 'volumetric distr noise noiseless whi5 adder 2',
          'volumetric distr noise noisy whi5 adder 3', 'Constant whi5 adder 4',
          'Noisy synthesis rate adder 5', '6', '7', '8', 'Constant Whi5 adder with no neg growth 9',
          'Noisy Whi5 adder with no neg growth 10', 'Constant Whi5 adder with const frac of division and no neg growth 11',
          'Constant Whi5 adder with const frac of division and neg growth 12', 'Const Whi5 adder and fraction with neg growth 13',
          'Const Whi5 adder and fraction with no neg growth 14', 'Noisy Whi5 adder and const fraction with neg growth 15',
          'Noisy Whi5 adder and const fraction with no neg growth 16', 'Noisy Integrating adder 17', 'Noisy Integrating adder with no neg growth 18']

par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 17), ('k_std', 0.2)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.2), ('delta', 1.0)])

models = [17, 18, 5, 10]
font = {'family': 'normal', 'weight': 'bold', 'size': 15}
plt.rc('font', **font)
celltype = ['m', 'd', 'p']
# figs = []
# for temp in models:
#     par1['modeltype'] = temp
#     figs.append(g.single_par_test(par1, val=1))
#     figs[-1].savefig('./lab_meeting_figures/model'+str(temp)+'_sample.eps', bbox_inches='tight', dpi=figs[-1].dpi,
#                      transparency=True)

g2_std = np.linspace(0.0, 0.2, 16)
g1_std = np.linspace(0.0, 0.2, 5)
num_rep = 3
X0, X1, X2, X3 = len(g2_std), len(g1_std), 2, num_rep
# a = np.zeros((X1, X0, X2, X3, 6, 3))

data = np.load('models'+str(models[0])+str(models[1])+'_script1.npy')
data = np.mean(data, axis=3)
par1['mothervals'] = True

for i in range(2):
    par1['modeltype'] = models[i]
    obs_new = np.empty([X1, X0, 6, 2])
    obs_new[:, :, :, :] = data[:, :, i, :, :2]
    figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))
    u = 1
    for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
        fig.savefig('./lab_meeting_figures/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                    + str(2) + '_cd'+str(67)+'_dstd'+str(2)+'V2.eps', bbox_inches='tight', dpi=fig.dpi)
        u += -1

# NEW PLOT

data = np.load('models'+str(models[2])+str(models[3])+'_script1.npy')
data = np.mean(data, axis=3)
par1['mothervals'] = True

for i in range(2):
    par1['modeltype'] = models[2+i]
    obs_new = np.empty([X1, X0, 6, 2])
    obs_new[:, :, :, :] = data[:, :, i, :, :2]
    figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))
    u = 1
    for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
        fig.savefig('./lab_meeting_figures/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                    + str(2) + '_cd'+str(67)+'_dstd'+str(2)+'V2.eps', bbox_inches='tight', dpi=fig.dpi)
        u += -1

# NEW PLOT

par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 1.0/0.67), ('td', 1.0), ('modeltype', 4), ('k_std', 0.2)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.0), ('delta', 1.0)])

data = np.load('models'+str(models[0])+str(models[1])+'_noiseless_script1.npy')
data = np.mean(data, axis=3)
# print data.shape
par1['mothervals'] = True

for i in range(2):
    par1['modeltype'] = models[i]
    obs_new = np.empty([X1, X0, 6, 2])
    obs_new[:, :, :, :] = data[:, :, i, :, :2]
    # print obs_new[0, :, 0, 1]
    figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))
    u = 1
    for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
        fig.savefig('./lab_meeting_figures/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                    + str(2) + '_cd'+str(67)+'_dstd'+str(0)+'.eps', bbox_inches='tight', dpi=fig.dpi)
        u += -1

# NEW PLOT

par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 1.0/0.67), ('td', 1.0), ('modeltype', 4), ('k_std', 0.2)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.0), ('delta', 1.0)])

data = np.load('models4'+'_noiseless_script1.npy')
data = np.mean(data, axis=3)
# print data.shape
par1['mothervals'] = True
i = 0
par1['modeltype'] = 4
obs_new = np.empty([X1, X0, 6, 2])
obs_new[:, :, :, :] = data[:, :, i, :, :2]
# print obs_new[0, :, 0, 1]
figs = g.test_function_syst(obs_new, par1, g1_std, g2_std*par1['td'], vec=range(len(g1_std)))
u = 1
for fig in figs:  # note that test_function_syst gives back daughter plot first, then mother plot.
    fig.savefig('./lab_meeting_figures/model'+str(par1['modeltype'])+'syst_test_'+celltype[u]+'_lstd'
                + str(2) + '_cd'+str(67)+'_dstd'+str(0)+'.eps', bbox_inches='tight', dpi=fig.dpi)
    u += -1

# New plot

cd = np.linspace(0.4, 0.8, 11)
par1 = dict([('g1_std', 0.0), ('g2_std', 0.15), ('g1_thresh_std', 0.0), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 100.0/0.67), ('td', 1.0), ('modeltype', 4), ('k_std', 0.0)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.2), ('delta', 100.0)])
# slopes = np.empty([len(cd), 6])
theory = np.empty([len(cd), 6])
models = [4, 9, 17, 18, 5, 10]
for i0 in range(len(cd)):
    par1['CD'] = cd[i0]
    for i1 in range(len(models)):
        par1['modeltype'] = models[i1]
        # obs = g.single_par_meas6(par1)
        # slopes[i0, i1] = obs[0, 1]
        theory[i0, i1] = g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['td']*par1['g2_std'])
# np.save('./lab_meeting_figures/modelcomparison', slopes)
slopes = np.load('./lab_meeting_figures/modelcomparison.npy')
values = range(len(models)+1)  # 2x since 2 models being compared
cmap = plt.get_cmap('gnuplot2')
cnorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
fig=plt.figure(figsize=[10, 10])
for i0 in range(len(models)/2):
    colorval = scalarmap.to_rgba(values[2*i0])
    if i0 == 1:
        label = ' $\sigma_i=0$, $\sigma_\lambda=0.2\lambda$, $\sigma_b=0.2t_d$, $\sigma_\Delta=0.2\Delta$'
    else:
        label = ' $\sigma_i=0$, $\sigma_\lambda=0.2\lambda$, $\sigma_b=0.2t_d$'
    plt.scatter(cd, slopes[:, 2 * i0], label='Model '+str(models[2 * i0]) + label, color=colorval)
    plt.plot(cd, theory[:, 2 * i0], color=colorval)
    colorval = scalarmap.to_rgba(values[2 * i0+1])
    plt.scatter(cd, slopes[:, 2 * i0 + 1], label='Model '+str(models[2 * i0 + 1])+label, color=colorval)
plt.xlim(xmin=0.38, xmax=0.82)
plt.title('Daughter Model comparison')
plt.xlabel('Budded phase duration $t_b/t_d$')
plt.ylabel('$V_d$ vs. $V_b$ regression slope')
plt.legend(loc=1)
fig.savefig('./lab_meeting_figures/model_comparison.eps', bbox_inches='tight', dpi=fig.dpi)
