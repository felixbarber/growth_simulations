#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
from mpi4py import MPI

# This script considers three different variants of the asymmetric yeast dilution model, and explores their phase spaces
# so that we can do a systematic analysis.
model_num = 3  # this is all that you need to change to switch between each of the three models.
models = [10, 18, 24, 26]

# Note
delta, td = 10.0, 1.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', None), ('dt', 0.01), ('td', td),
                 ('g1_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('K', None), ('CD', None), ('r', None),
                 ('r_std', None), ('g2_std', None), ('d_std', None), ('k_std', None), ('g1_thresh_std', None)])
# Model list. Note that we assume no negative growth for all.
models_descr = ['Noisy synthesis rate G2 timer', 'Noisy integrator G2 timer', 'Noisy integrator r sensor']

num_rep = 3
L = 6  # number of discrete points to break space into
g1_thresh_std = np.linspace(0.0, 0.3, L)

# cd = np.linspace(0.45, 0.7, 6)  # Original ranges
# g2_std = np.linspace(0.0, 0.3, L)
# l_std = np.linspace(0.0, 0.3, L)
# r_std = np.linspace(0.0, 0.3, L)
# r = np.linspace(0.45, 0.7, 6)

cd = np.linspace(0.585, 0.765, L)
g2_std = np.linspace(0.0, 0.16, L)

l_std = np.linspace(0.0, 0.3, L)
r_std = np.linspace(0.0, 0.3, L)
r = np.linspace(0.5, 0.7, 6)

k_std = np.linspace(0.0, 0.3, L)
d_std = np.linspace(0.0, 0.3, L)

fp = [g1_thresh_std, cd, g2_std, l_std, r, r_std, k_std, d_std]
fv = ['g1_thresh_std', 'CD', 'g2_std', 'l_std', 'r', 'r_std', 'k_std', 'd_std']
# This details all the variables in the above models. Which variables are considered will differ slightly per model.
model_vars = [[0, 1, 2, 3, 6], [0, 1, 2, 3, 7], [0, 4, 5, 7], [0, 4, 5, 6]]

num_celltype = 2
num_sims = 2  # number of different simulations to be run
num_meas = 3  # number of different kinds of measurements

par_vals['modeltype'] = models[model_num]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    # Y gives the length of variation for each index.
    Y = 1
    Z = []  # Z is a list with the shape of the array that this needs to map to.
    for i0 in model_vars[model_num]:
        Y = Y * len(fp[i0])  # gives the range of lengths for each modeltype
        Z.append(len(fp[i0]))
    # Here X gives the basic structure of the matrix this maps into.
    X = Z + [num_rep, num_celltype, num_sims, num_meas]
    a = np.zeros(X)
    if rank == 0:
        print 'Total rep number = {0}'.format(Y*num_rep), ' shape:', X, ' model {0}'.format(par_vals['modeltype'])
    if rank == 0:
        print 'Setup matrix'
    dx = Y / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = Y
    if model_num in [0, 1]:  # have to do it this way since the matrices have diff dim no. for diff. models
        for i0 in xrange(start, stop):  # varying the different growth conditions
            temp = np.unravel_index(i0, tuple(Z))  # gives the indices of this in phase space as implied by model_vars
            par1 = par_vals
            for i1 in range(len(temp)):  # set the values for each of the different indices in this range.
                par1[fv[model_vars[model_num][i1]]] = fp[model_vars[model_num][i1]][temp[i1]]
            if model_num == 0:
                par1['K'] = delta/par1['CD']
            for i1 in range(num_rep):
                # tic1 = time.clock()
                c = []
                # print par1
                temp1 = g.discr_gen(par1)
                c.append(temp1)
                del temp1
                # This will initialize the subsequent simulations for this model
                temp2 = g.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
                # initial pop seeded from c
                temp1, obs1 = g.discr_time_1(par1, temp2)
                c.append(temp1)
                del temp1, temp2
                for i2 in range(num_celltype):  # celltype
                    for i3 in range(num_sims):  # sim number
                        x1 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2]
                        x2 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2 and obj.wd / obj.vd < 1.0]
                        x3 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2 and obj.wb / obj.vb < 1.0]
                        y1 = [obj.vd for obj in c[i3][1000:] if obj.isdaughter == i2]
                        val1 = scipy.stats.linregress(x1, y1)
                        a[temp[0], temp[1], temp[2], temp[3], temp[4], i1, i2, i3, 0] = val1[0]  # slope result
                        a[temp[0], temp[1], temp[2], temp[3], temp[4], i1, i2, i3, 1] = len(x2) * 100.0 / len(x1)
                        # % with low div conc result
                        a[temp[0], temp[1], temp[2], temp[3], temp[4], i1, i2, i3, 2] = len(x3) * 100.0 / len(x1)
                        # % with low birth conc result
                del c, obs1
                # print rank, "time taken", time.clock()-tic1
                # print a[temp[0], temp[1], temp[2], temp[3], temp[4], i1, 1, 1, :]
                # print par1
                # exit()
            if np.mod(i0-start, 1000) == 0:
                print 'I am {0} and I have done {1} ranges'.format(str(rank), str(i0-start))
    else:
        for i0 in xrange(start, stop):  # varying the different growth conditions
            temp = np.unravel_index(i0, tuple(Z))  # gives the indices of this in phase space as implied by model_vars
            par1 = par_vals
            for i1 in range(len(temp)):  # set the values for each of the different indices in this range.
                par1[fv[model_vars[model_num][i1]]] = fp[model_vars[model_num][i1]][temp[i1]]
            if model_num == 3:
                # we reset K here so that the cell size distribution remains within the same range for different noise
                # and r values.
                par1['K'] = np.log(2.0)/(np.log(1.0 + par1['r']) - 0.5 * par1['r_std'] ** 2 / (1 + par1['r']) ** 2)
            for i1 in range(num_rep):
                # tic1 = time.clock()
                c = []

                temp1 = g.discr_gen(par1)
                c.append(temp1)
                del temp1
                # This will initialize the subsequent simulations for this model
                temp2 = g.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
                # initial pop seeded from c
                temp1, obs1 = g.discr_time_1(par1, temp2)
                c.append(temp1)
                del temp1, temp2
                for i2 in range(num_celltype):  # celltype
                    for i3 in range(num_sims):  # sim number
                        x1 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2]
                        x2 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2 and obj.wd / obj.vd < 1.0]
                        x3 = [obj.vb for obj in c[i3][1000:] if obj.isdaughter == i2 and obj.wb / obj.vb < 1.0]
                        y1 = [obj.vd for obj in c[i3][1000:] if obj.isdaughter == i2]
                        val1 = scipy.stats.linregress(x1, y1)
                        a[temp[0], temp[1], temp[2], temp[3], i1, i2, i3, 0] = val1[0]  # slope result
                        a[temp[0], temp[1], temp[2], temp[3], i1, i2, i3, 1] = len(x2) * 100.0 / len(x1)
                        # % with low div conc result
                        a[temp[0], temp[1], temp[2], temp[3], i1, i2, i3, 2] = len(x3) * 100.0 / len(x1)
                        # % with low birth conc result
                del c, obs1
                # print rank, "time taken", time.clock()-tic1
                # print a[temp[0], temp[1], temp[2], temp[3], i1, 1, 1, :]
                # exit()
            if np.mod(i0 - start, 1000) == 0:
                print 'I am {0} and I have done {1} ranges'.format(str(rank), str(i0 - start))
    # Now everyone has made part of the matrix. Send to one processor. Many ways to do this. Broadcast, sendrecv etc
    if rank != 0:
        comm.send(a, dest=0)
    if rank == 0:
        new_grid = np.zeros(np.shape(a))
        new_grid += a
        for p in range(1, size):
            print 'I am 0 and I got from ', p
            new_grid += comm.recv(source=p)
    comm.barrier()
    if rank == 0:
        print 'Time taken =', time.clock()-tic
        # plt.imshow(new_grid)
        # plt.show()
        np.save('April17_simulation_dilution_asymmetric_model_{0}_revised'.format(par_vals['modeltype']), new_grid)

# Model 10 took 22420.046008s
