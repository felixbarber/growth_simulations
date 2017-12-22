#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_accumulator_asymmetric as h
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import pandas as pd
import seaborn as sns
# Setting SNS styles
sns.set_context('paper', font_scale=2.2)
sns.set_style('ticks')

par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3), ('g2_std', 0.02), ('l_std', 0.02),
            ('CD', 0.65)])

# plotter


models = [1, 2, 3, 4]
Celltype = ['Mothers', 'Daughters']
# for i1 in models:
#     par1['modeltype'] = i1
#     c = h.discr_gen(par1)
#     for i0 in range(2):
#         x = [obj.vb for obj in c if obj.isdaughter == i0]
#         y = [obj.vd for obj in c if obj.isdaughter == i0]
#         temp = scipy.stats.linregress(x, y)
#         print 'Modeltype', i1, "celltype", Celltype[i0], temp[:2]

# for i1 in [3, 4]:
#     par1['modeltype'] = i1
#     c = h.discr_gen(par1)
#     x = np.asarray([obj.vi for obj in c[1000:] if obj.isdaughter == 0])
#     y = np.asarray([obj.vi for obj in c[1000:] if obj.isdaughter == 1])
#     print 'Modeltype', i1, 0.5*(np.mean(x*x)+np.mean(y*y)), h.vivi(par1, celltype=2)

# for i1 in [3, 4]:
#     par1['modeltype'] = i1
#     c = h.discr_gen(par1)
#     for i2 in range(2):
#         x = np.asarray([obj.vb for obj in c[1000:] if obj.isdaughter == i2])
#         y = np.asarray([obj.vd for obj in c[1000:] if obj.isdaughter == i2])
#         vals = scipy.stats.linregress(x, y)
#         print 'Modeltype', i1, "celltype", i2, "slope", vals[0], h.slope_vbvd(par1, celltype=i2)
#         # print 'Modeltype', i1, "celltype", i2, "<vb>", np.mean(x), h.vb(par1, celltype=i2)
#         # print 'Modeltype', i1, "celltype", i2, "<vd>", np.mean(y), h.vd(par1, celltype=i2)
#         # print 'Modeltype', i1, "celltype", i2, "<vbvd>", np.mean(x*y), h.vdvb(par1, celltype=i2)
#         # print 'Modeltype', i1, "celltype", i2, "<vbvb>", np.mean(x * x), h.vbvb(par1, celltype=i2)
#     x = np.asarray([obj.vb for obj in c[1000:]])
#     y = np.asarray([obj.vd for obj in c[1000:]])
#     vals = scipy.stats.linregress(x, y)
#     print 'Modeltype', i1, "celltype", 2, "slope", vals[0], h.slope_vbvd(par1, celltype=2)
modeltype = 'Yeast accumulator model, with noise in $r$ '
lab_0 = list(['Accumulator Negative growth', 'Accumulator No Negative growth'])
lab_1 = ['Mothers', 'Daughters']
lab_2 = ['Discrete Gen', 'Discrete time']
filenames = ['March17_accumulator1_V0']
for i1 in range(len(filenames)):
    temp_name = filenames[i1]
    path = './'+temp_name
    df = pd.read_pickle(path)
    # print v.shape, v[10], df.shape
    # dataframe column labels
    lab = ['full label', 'Model', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
    for i0 in range(2):
        fig = plt.figure(figsize=[15, 6])
        sns.boxplot(data=df[df['Obs. type'].isin([lab_2[i0]]) & df['Model'].isin([lab_0[1]])], x='Model', y=lab[-1], hue='Cell type',
                       palette="Set3")
        plt.title(modeltype+lab_2[i0]+' '+lab[-1])
        fig.savefig('../data_analysis/pandas/March17_accumulator_model_'+str(i1)+'_meastype_' + str(i0) + 'NNG.eps', bbox_inches='tight', dpi=fig.dpi)
        del fig
    for i0 in range(2):
        fig = plt.figure(figsize=[15, 6])
        sns.boxplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Model', y=lab[-1], hue='Cell type',
                       palette="Set3")
        plt.title(modeltype+lab_2[i0]+' '+lab[-1])
        fig.savefig('../data_analysis/pandas/March17_accumulator_model_'+str(i1)+'_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
        del fig
    del df

# Now we create the heatmaps that will be used in the paper for the asymmetric yeast section.

data = np.load(filenames[0]+'.npy')
delta = 10.0
par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3)])
r = np.linspace(0.45, 0.75, 8)
r_std = np.linspace(0.0, 0.28, 8)
g1_thresh_std = np.linspace(0.0, 0.28, 8)
models = [3, 4]
num_rep = 5
num_celltype = 2
num_sims = 2
num_meas = 2
# X = [len(r), len(r_std), len(g1_thresh_std), len(models), num_rep, num_celltype, num_sims, num_meas]
celltype = ['Mothers', 'Daughters']
model_descr = ['Cells can shrink', 'Cells cannot shrink']
obs_type = ['$V_b$ $V_d$ slope', '% with $A_b><\Delta>$']
lab = ['$\sigma_{r}/r$', '$r$']
g1_var = [0, 5]
for i0 in range(len(models)):
    for i1 in range(num_celltype):
        for i2 in range(num_sims):
            for i3 in range(num_meas):
                for i4 in g1_var:
                    obs = np.mean(data[:, :, i4, i0, :, i1, i2, i3], axis=-1)
                    fig = plt.figure(figsize=[8, 8])
                    ax = plt.subplot(1, 1, 1)
                    temp = celltype[i1]+', '+' $\sigma_i/<\Delta>=$'+str(np.round(g1_thresh_std[i4], 2))
                    ax = g.heat_map(obs, r, r_std, ax, xlabel=lab[0], ylabel=lab[1], title=temp, fmt='.2g')
                    fig.savefig('./March17_accumulator0'+'_celltype_'+str(i1)+'_model_' + str(models[i0]) + '_meas_' +
                                str(i3) + '_sim_' + str(i2)+'_si_'+str(i4) + '.eps', bbox_inches='tight', dpi=fig.dpi)
                    del fig

