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
filename = 'Feb17_test_models3_model16_V2'
path = './'+filename
df = pd.read_pickle(path)

# dataframe column labels
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
for i0 in range(3):
    fig = plt.figure(figsize=[15, 6])
    sns.violinplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                   palette="Set3")
    plt.title(lab_2[i0]+' '+lab[-1]+' for different growth media and celltypes')
    fig.savefig('../data_analysis/pandas/Dip_slopes_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig

del df

# loading our data frame
filename = 'Feb17_test_models3_model16_V2_si_0'
path = './'+filename
df = pd.read_pickle(path)

# dataframe column labels
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
for i0 in range(3):
    fig = plt.figure(figsize=[15, 6])
    sns.boxplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                   palette="Set3")
    plt.title(lab_2[i0]+' '+lab[-1]+' for different growth media and celltypes')
    fig.savefig('../data_analysis/pandas/Dip_slopes_meastype_' + str(i0) + '_si_0.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig

del df

# loading our data frame
filename = 'Feb17_test_models3_model16_old'
path = './'+filename
df = pd.read_pickle(path)

# dataframe column labels
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
for i0 in range(3):
    fig = plt.figure(figsize=[15, 6])
    sns.violinplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                   palette="Set3")
    plt.title(lab_2[i0]+' '+lab[-1]+' for different growth media and celltypes')
    fig.savefig('../data_analysis/pandas/Dip_slopes_old_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig

# loading our data frame
filename = 'Feb17_test_models3_model16_V3'
path = './'+filename
df = pd.read_pickle(path)

# dataframe column labels
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
for i0 in range(3):
    fig = plt.figure(figsize=[15, 6])
    sns.violinplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                   palette="Set3")
    plt.title(lab_2[i0]+' '+lab[-1]+' for different growth media and celltypes')
    fig.savefig('../data_analysis/pandas/Dip_slopes_V3_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig

# loading our data frame
filename = 'Feb17_test_models3_model16_V3'+'_constr'
path = './'+filename
df = pd.read_pickle(path)

# dataframe column labels
lab = ['full label', 'Growth Medium', 'Cell type', 'Obs. type', '$V_{d}$ $V_{b}$ Slope']
for i0 in range(3):
    fig = plt.figure(figsize=[15, 6])
    sns.violinplot(data=df[df['Obs. type'].isin([lab_2[i0]])], x='Growth Medium', y=lab[-1], hue='Cell type',
                   palette="Set3")
    plt.title(lab_2[i0]+' '+lab[-1]+' for different growth media and celltypes')
    fig.savefig('../data_analysis/pandas/Dip_slopes_V3_constr_meastype_' + str(i0) + '.eps', bbox_inches='tight', dpi=fig.dpi)
    del fig
