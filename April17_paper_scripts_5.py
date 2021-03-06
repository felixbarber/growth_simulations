#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_accumulator_asymmetric as h
import time
from scipy import stats
import scipy


# This script will be used as a first tester point for variability in the observed slope for a noisy adder model in
# which there is  noise only in r, not in the budded time or in the growth rate lambda.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 3 parallel discretized time simulations.
# We then look at the mean slope for each parameter value, and check the result

delta = 10.0

par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', 10.0), ('g1_thresh_std', 0.3)])

L = 7
r = np.linspace(0.45, 1.0, 12)
r_std = np.linspace(0.0, 0.3, L)
g1_thresh_std = np.linspace(0.0, 0.3, L)

models = [3, 4]
# models = [4]
num_rep = 20
num_celltype = 3
num_sims = 2
num_meas = 2
timer = 7.6

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'reps: ', num_rep, 'meas: ', num_meas
        print 'expected time taken=', timer * len(models) * len(r) * L ** 2 * num_rep * 1.0 / (3600.0 * 8)
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [len(r), len(r_std), len(g1_thresh_std), len(models), num_rep, num_celltype, num_sims, num_meas]
    a = np.zeros(X)

    dx = num_rep / size
    rem = np.mod(num_rep, size)
    if rank >= size - rem:  # this makes sure that it distributes the remainder as equally as possible.
        start = dx * (size - rem) + (dx + 1) * (rank + rem - size)
        stop = start + dx + 1
    else:
        start = dx * rank
        stop = start + dx
    if rank == size - 1:
        stop = num_rep
    print 'I am {0}, my start is {1}, stop {2}'.format(str(rank), str(start), str(stop))
    pars = [r, r_std, g1_thresh_std, models]
    vals = ['r', 'r_std', 'g1_thresh_std', 'modeltype']
    for i0 in range(X[0]):  # varying the different growth conditions
        par1[vals[0]] = pars[0][i0]
        for i1 in range(X[1]):
            par1[vals[1]] = pars[1][i1]
            for i2 in range(X[2]):
                par1[vals[2]] = pars[2][i2]
                for i3 in range(X[3]):
                    par1[vals[3]] = pars[3][i3]
                    for i4 in xrange(start, stop):
                        # tic = time.clock()
                        c = []
                        temp1 = h.discr_gen(par1)
                        c.append(temp1)
                        del temp1
                        # This will initialize the subsequent simulations for this model
                        temp = h.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
                        # initial pop seeded from c
                        temp1, obs1 = h.discr_time_1(par1, temp)
                        c.append(temp1)
                        del temp1, temp
                        for i5 in range(X[5]):  # celltype
                            if i5 <= 1:
                                for i6 in range(X[6]):  # sim number
                                    x1 = [obj.vb for obj in c[i6][1000:] if obj.isdaughter == i5]
                                    x2 = [obj.vb for obj in c[i6][1000:] if obj.isdaughter == i5 and obj.wb > par1['delta']]
                                    y1 = [obj.vd for obj in c[i6][1000:] if obj.isdaughter == i5]
                                    val1 = scipy.stats.linregress(x1, y1)
                                    a[i0, i1, i2, i3, i4, i5, i6, 0] = val1[0]  # slope result
                                    a[i0, i1, i2, i3, i4, i5, i6, 1] = len(x2)*100.0/len(x1)  # % with high birth conc
                                # % with low div conc result
                            else:
                                for i6 in range(X[6]):  # sim number
                                    x1 = [obj.vb for obj in c[i6][1000:]]
                                    x2 = [obj.vb for obj in c[i6][1000:] if obj.wb > par1['delta']]
                                    y1 = [obj.vd for obj in c[i6][1000:]]
                                    val1 = scipy.stats.linregress(x1, y1)
                                    a[i0, i1, i2, i3, i4, i5, i6, 0] = val1[0]  # slope result
                                    a[i0, i1, i2, i3, i4, i5, i6, 1] = len(x2) * 100.0 / len(
                                        x1)  # % with high birth conc
                                    # % with low div conc result
                        # print str(rank)
                        # del c, obs1
                        # print rank, "r=", par1['r'], "time taken", time.clock()-tic
                        # print a[i0, i1, i2, i3, i4, :, 1, 0]
                        # exit()
            print 'I am {0} and I have done {1} ranges'.format(str(rank), str(i1))
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
        np.save('April17_paper_scripts_5_v1', new_grid)

# took 5600s.
# April17_paper_scripts_5_v0 has no negative growth whatsoever in models 2 and 4 and 5 reps.. Took 9053s.
# April17_paper_scripts_5_v1 has no negative growth whatsoever in models 2 and 4 and has 20 reps. Also stores info for
# whole population.