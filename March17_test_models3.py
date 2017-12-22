#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
import pandas as pd
import seaborn as sns

# Setting SNS styles
sns.set_context('paper', font_scale=2.2)
sns.set_style('ticks')

lab_0 = list(['Glucose', 'Galactose', 'Glycerol', 'Low Glucose', 'Raffinose'])
lab_1 = ['Mothers', 'Daughters']
lab_2 = ['Discrete Gen', 'Discrete time', 'Theory']

# loading our data frame
filename = 'March17_test_models1_model16_V0'
modeltype = 'Noisy S.R. '
filenames = [filename]
vec = [[0.0, 0.1], [0.0, 0.2]]
descr = ['$\sigma_{\lambda}/\lambda \in [0.0, 0.1]$, $\sigma_{K}/K \in [0.0, 0.2]$']
for i0 in range(2):
    for i1 in range(2):
        temp = filename+'_s_i'+str(i0)+'_s_k'+str(i1)
        filenames.append(temp)
        descr.append('$\sigma_{i}/<\Delta>=$'+str(vec[0][i0])+', $\sigma_{K}/K =$'+str(vec[1][i1]))
        del temp

for i1 in range(len(filenames)):
    temp_name = filenames[i1]
    path = './'+temp_name
    df = pd.read_pickle(path)

    # dataframe column labels
    lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
    for i0 in range(3):
        fig = plt.figure(figsize=[15, 6])
        sns.boxplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                       palette="Set3")
        plt.title(modeltype+lab_2[i0]+' '+lab[-1]+' for '+descr[i1])
        fig.savefig('../data_analysis/pandas/March17_model16_set_'+str(i1)+'_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
        del fig
    del df

# noiseless adder model
# loading our data frame
filename = 'March17_test_models1_model22_V0'
modeltype = 'Noiseless int. '
filenames = [filename]
vec = [[0.0, 0.1], [0.0, 0.2]]
descr = ['$\sigma_{\lambda}/\lambda \in [0.0, 0.1]$']
for i0 in range(2):
    temp = filename+'_s_i'+str(i0)
    filenames.append(temp)
    descr.append('$\sigma_{i}/<\Delta>=$'+str(vec[0][i0]))
    del temp

for i1 in range(len(filenames)):
    temp_name = filenames[i1]
    path = './'+temp_name
    df = pd.read_pickle(path)

    # dataframe column labels
    lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
    for i0 in range(3):
        fig = plt.figure(figsize=[15, 6])
        sns.boxplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                       palette="Set3")
        plt.title(modeltype+lab_2[i0]+' '+lab[-1]+' for '+descr[i1])
        fig.savefig('../data_analysis/pandas/March17_model22_set_'+str(i1)+'_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
        del fig
    del df