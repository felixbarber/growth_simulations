#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import growth_simulation_accumulator_asymmetric as h
import time
import scipy
from scipy import stats
import os.path


font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

delta = 10.0

par_vals = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
                 ('r_std', 0.02), ('delta', 10.0), ('d_std', 0.0), ('g1_thresh_std', 0.02), ('g1_std', 0.0),
                 ('g1_delay', 0.0)])

models = [23, 24, 3, 4]
L = 17
r = np.linspace(0.6, 1.0, L)
vals = np.empty([2, L, 4])
celltype=['Mothers', 'Daughters']
modeldescr=['Inhib. dil. 23', 'Inhib. dil. 24', 'Init. acc. 3', 'Init. acc. 4']
path = './April17_paper_scripts_r_var'
path1 = './April17_paper_scripts_r_var_1'
markertype = ['^', 'o', 's', 'D']

while not os.path.isfile(path+'.npy'):  # ensures that this dataset is there
    for i0 in range(len(r)):
        par_vals['r'] = r[i0]
        for i2 in range(4):
            par_vals['modeltype'] = models[i2]
            if i2 < 2:
                temp = g.discr_gen(par_vals)
                temp1 = g.starting_popn_seeded([obj for obj in temp if obj.exists], par_vals)
                temp2 = g.discr_time_1(par_vals, temp1)
            else:
                temp = h.discr_gen(par_vals)
                temp1 = h.starting_popn_seeded([obj for obj in temp if obj.exists], par_vals)
                temp2 = h.discr_time_1(par_vals, temp1)
            for i1 in range(2):
                x = [obj.vb for obj in temp2[0][1000:] if obj.isdaughter == i1]
                y = [obj.vd for obj in temp2[0][1000:] if obj.isdaughter == i1]
                temp3 = scipy.stats.linregress(x, y)
                vals[i1, i0, i2] = temp3[0]
            del temp, temp1, temp2
        print 'Done r={0}'.format(str(r[i0]))
    np.save(path, vals)
    del vals

data = np.load(path+'.npy')
fig = plt.figure(figsize=[12, 5])
for i0 in range(2):
    ax = plt.subplot(1, 2, 1+i0)
    for i1 in range(len(models)):
        ax.plot(r, data[i0, 8:, i1], markertype[i1], markersize=5.0, label=str(modeldescr[i1]))
    plt.legend()
    ax.set_xlabel('$r$')
    ax.set_ylabel('$V_b$ $V_d$ slope')
    ax.set_title(celltype[i0])
fig.savefig(path+'.eps', bbox_inches='tight', dpi=fig.dpi)

par_vals['r_std'] = 0.01
par_vals['g1_thresh_std'] = 0.01
while not os.path.isfile(path1+'.npy'):  # ensures that this dataset is there
    for i0 in range(len(r)):
        par_vals['r'] = r[i0]
        for i2 in range(4):
            par_vals['modeltype'] = models[i2]
            if i2 < 2:
                temp = g.discr_gen(par_vals)
                temp1 = g.starting_popn_seeded([obj for obj in temp if obj.exists], par_vals)
                temp2 = g.discr_time_1(par_vals, temp1)
            else:
                temp = h.discr_gen(par_vals)
                temp1 = h.starting_popn_seeded([obj for obj in temp if obj.exists], par_vals)
                temp2 = h.discr_time_1(par_vals, temp1)
            for i1 in range(2):
                x = [obj.vb for obj in temp2[0][1000:] if obj.isdaughter == i1]
                y = [obj.vd for obj in temp2[0][1000:] if obj.isdaughter == i1]
                temp3 = scipy.stats.linregress(x, y)
                vals[i1, i0, i2] = temp3[0]
            del temp, temp1, temp2
        print 'Done r={0}'.format(str(r[i0]))
    np.save(path1, vals)
    del vals

data = np.load(path1+'.npy')
fig = plt.figure(figsize=[12, 5])
for i0 in range(2):
    ax = plt.subplot(1, 2, 1+i0)
    for i1 in range(len(models)):
        ax.plot(r, data[i0, :, i1], markertype[i1], markersize=5.0, label=str(modeldescr[i1]))
    plt.legend()
    ax.set_xlabel('$r$')
    ax.set_ylabel('$V_b$ $V_d$ slope')
    ax.set_title(celltype[i0])
fig.savefig(path1+'.eps', bbox_inches='tight', dpi=fig.dpi)