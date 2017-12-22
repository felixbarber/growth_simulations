#!/usr/bin/env python

import os
import cluster_dilution_source as g
import time
from scipy import stats
import scipy
import numpy as np

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

erase = 1  # variable to state whether this simulation should overwrite any existing files
hard_erase = 1  # If this is true, previous files will all be completely overwritten.
size = 200  # starts from 1.
rank = int(os.environ["PARAM1"]) - 1  # starts from zero. Imports the index tracking the number of iterations

# ------------------------------------------------  specific to individual scripts

path1 = './data/inh_yeast_asymm_{0}.npy'.format(rank)  # path where the calculated values will be saved
path2 = './data/prog_{0}.npy'.format(rank)  # path where progress made is recorded.


delta = 10.0
par1 = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
                ('num_gen', 9), ('modeltype', 24), ('delta', delta)])

model_descr = ['Noisy integrator no neg growth']
r = np.linspace(0.5, 0.7, 3)
r_std = np.linspace(0.0, 0.3, 31)
d_std = np.linspace(0.0, 0.3, 31)
g1_thresh_std = np.linspace(0.0, 0.3, 4)
num_rep = 20
num_celltype = 2
num_sims = 2
num_meas = 3

X = [len(r), len(r_std), len(g1_thresh_std), len(d_std), num_rep, num_celltype, num_sims, num_meas]
# Celltype must be in 5, num_sims must be in 6, and num_meas in 7
pars = [r, r_std, g1_thresh_std, d_std]
vals = ['r', 'r_std', 'g1_thresh_std', 'd_std']

b = np.zeros(X[:5])
# ------------------------------------------------

a = np.zeros(X)
temp_v0 = len(b.flatten())
dx = temp_v0 / (size-1)
rem = np.mod(temp_v0, size-1)

start = dx * rank
stop = dx * (rank + 1)
if rank == size-1:
    stop = temp_v0

if not hard_erase:
    if os.path.isfile(path1) and np.sum(np.load(path1).shape == (np.asarray(X)-1)) == 0:  # checks that this file exists,
        # and if it does, that the shape is the same (controls for shorter trials).
        # allows us to pick up where we left off just in case it terminated early
        if os.path.isfile(path2):
            temp_v2 = np.load(path2)
            start = temp_v2[0]
            a = np.load(path1)
            print "Restarted from: ", start
        elif erase:
            print "Erase. Will erase previously existent file."
        else:
            raise ValueError('Change erase settings')
else:
    print "Hard erase. Will erase previously existent file."

print 'I am {0}, my start is {1}, stop {2}, total number {3}'.format(str(rank), str(start), str(stop), str(stop-start))

for index in xrange(start, stop):
    temp_v1 = np.unravel_index(index, b.shape)
    for i0 in range(len(pars)):  # assumes that the variables in pars come first in b.
        par1[vals[i0]] = pars[i0][temp_v1[i0]]  # sets the variables to that specified by the parameter set given

    # tic = time.clock()

    c = []
    temp1 = g.discr_gen(par1)
    c.append(temp1)
    del temp1
    # This will initialize the subsequent simulations for this model
    temp = g.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
    # initial pop seeded from c
    temp1, obs1 = g.discr_time_1(par1, temp)
    c.append(temp1)
    del temp1, temp
    for i5 in range(X[5]):  # celltype
        for i6 in range(X[6]):  # sim number
            x1 = [obj.vb for obj in c[i6] if obj.isdaughter == i5]
            x2 = [obj.vb for obj in c[i6] if obj.isdaughter == i5 and obj.wd/obj.vd < 1.0]
            x3 = [obj.vb for obj in c[i6] if obj.isdaughter == i5 and obj.wb / obj.vb < 1.0]
            y1 = [obj.vd for obj in c[i6] if obj.isdaughter == i5]
            val1 = scipy.stats.linregress(x1, y1)
            a[temp_v1[0], temp_v1[1], temp_v1[2], temp_v1[3], temp_v1[4], i5, i6, 0] = val1[0]  # slope result
            a[temp_v1[0], temp_v1[1], temp_v1[2], temp_v1[3], temp_v1[4], i5, i6, 1] = len(x2)*100.0/len(x1)  # % with low div conc result
            a[temp_v1[0], temp_v1[1], temp_v1[2], temp_v1[3], temp_v1[4], i5, i6, 2] = len(x3) * 100.0 / len(x1)
            # % with low div conc result
    del c

    # print rank, "r=", par1['r'], "time taken", time.clock()-tic
    # print a[i0, i1, i2, i3, i4, 0, 1, :]
    # exit()

    if np.mod(index-start, 50) == 0:
        print 'I am {0} and I have done {1} ranges'.format(rank, index-start)
    if np.mod(index-start, 10) == 0:
        np.save(path1, a)  # overwrite every 10 saves
        temp_v3 = np.array([index])
        np.save(path2, temp_v3)
        print 'I am {0} and I have saved {1} ranges'.format(rank, index-start)
        del temp_v3

np.save(path1, a)  # final save
temp_v3 = np.array([stop])
np.save(path2, temp_v3)
del temp_v3

# print 'time taken:', time.clock()-tic
# Time taken for 80 repeats: 778s. ~ 9.7s per run.
