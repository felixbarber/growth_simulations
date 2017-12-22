#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting SNS styles
sns.set_context('paper', font_scale=2.2)
sns.set_style('ticks')

# models = [10, 18, 24, 26]
models = [10, 18, 24]
print pd.__version__
# fp = [g1_thresh_std, cd, g2_std, l_std, r, r_std, k_std, d_std]
fv = ['g1_thresh_std', 'CD', 'g2_std', 'l_std', 'r', 'r_std', 'k_std', 'd_std']
# This details all the variables in the above models. Which variables are considered will differ slightly per model.
model_vars = [[0, 1, 2, 3, 6], [0, 1, 2, 3, 7], [0, 4, 5, 7], [0, 4, 5, 6]]

models_descr = ['Noisy synthesis rate G2 timer', 'Noisy integrator G2 timer', 'Noisy integrator r sensor']
# Here we populate our dataframe based on the simple outputs from the initial matplotlib files.

celltype = ['Mothers', 'Daughters']
obs_types = ['$V_b$ $V_d$ slope', '$% Popn [W]_d<1.0$', '$% Popn [W]_b<1.0$']
meas_types = ['max', 'min']

# In this we produce plots of the max and min variability in observed slope for varying each kind of variable while
# keeping all others fixed.

labels = ['type', 'Model Parameter $y$', 'celltype', '$V_b$ $V_d$ slope', '$% Popn [W]_d<1.0$', '$% Popn [W]_b<1.0$']
# pd dataframe column labels
num = ['(A) ', '(B) ', '(C) ', '(D) ']
for i0 in range(len(models)):
    # path = './April17_simulation_dilution_asymmetric_model_{0}'.format(models[i0])
    path = './April17_simulation_dilution_asymmetric_model_{0}_revised'.format(models[i0])
    df = pd.read_pickle(path)
    for i1 in range(len(celltype)):
        # for i2 in range(len(obs_types)):
        i2=0
        fig = plt.figure(figsize=[15, 6])
        sns.boxplot(data=df[df['celltype'].isin([celltype[i1]])], x='Model Parameter $y$', y=obs_types[i2], hue='type'
                       ,palette="Set3")
        plt.ylabel(obs_types[i2]+' variation')
        plt.title(num[i0]+celltype[i1]+' '+obs_types[i2]+' variation for different variables')
        # fig.savefig('../data_analysis/pandas/April17_simulation_dilution_asymmetric_2_model_{0}_celltype_{1}_obs_{2}.eps'.format(
        #     str(models[i0]), str(i1), str(i2)), bbox_inches='tight', dpi=fig.dpi)
        fig.savefig(
            '../data_analysis/pandas/April17_simulation_dilution_asymmetric_2_revised_model_{0}_celltype_{1}_obs_{2}.eps'.format(
                str(models[i0]), str(i1), str(i2)), bbox_inches='tight', dpi=fig.dpi)
        del fig

for i1 in range(len(celltype)):
    fig = plt.figure(figsize=[15, 25])
    for i0 in range(len(models)):
        ax=plt.subplot(len(models), 1, i0+1)
        # path = './April17_simulation_dilution_asymmetric_model_{0}'.format(models[i0])
        path = './April17_simulation_dilution_asymmetric_model_{0}_revised'.format(models[i0])
        df = pd.read_pickle(path)
        # for i2 in range(len(obs_types)):
        i2=0
        sns.boxplot(data=df[df['celltype'].isin([celltype[i1]])], x='Model Parameter $y$', y=obs_types[i2], hue='type'
                       ,palette="Set3")
        ax.set_ylabel(obs_types[i2] + ' variation')
        plt.title(num[i0]+celltype[i1]+' '+obs_types[i2]+' variation for different variables')
    # fig.savefig('../data_analysis/pandas/April17_simulation_dilution_asymmetric_2_celltype_{0}_obs_{1}.eps'.format(str(i1), str(i2)), bbox_inches='tight', dpi=fig.dpi)
    fig.savefig(
        '../data_analysis/pandas/April17_simulation_dilution_asymmetric_2_revised_celltype_{0}_obs_{1}.eps'.format(str(i1),
                                                                                                           str(i2)),
        bbox_inches='tight', dpi=fig.dpi)

    del fig

