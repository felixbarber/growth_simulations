#!/usr/bin/env python

import numpy as np
import growth_simulation_accumulator_asymmetric as h
import time
from scipy import stats
import scipy
from mpi4py import MPI

par_vals = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 1.0), ('r_std', 0.0), ('delta', 10.0), ('g1_thresh_std', 0.0)])

L = 7
models = [3, 4]
g1_thresh_std = np.linspace(0.0, 0.3, L)

vals = ['g1_thresh_std']
pars = [g1_thresh_std]
num_rep = 20  # number of repeats for each condition
num_celltype = 2
num_sims = 2  # number of different simulations to be run
num_meas = 2  # number of different kinds of measurements

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [num_rep, len(models), L, num_celltype, num_sims, num_meas]
    a = np.zeros(X)
    if rank == 0:
        print 'Setup matrix'
    dx = X[0] / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X[0]
    for i0 in xrange(start, stop):  # varying the different growth conditions
        for i1 in range(X[1]):
            par1 = par_vals
            par1['modeltype'] = models[i0]
            for i1 in range(X[1]):
                par1[vals[0]] = pars[0][i1]
                for i3 in range(X[3]):
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
                    for i4 in range(X[4]):  # celltype
                        for i5 in range(X[5]):  # sim number
                            x1 = [obj.vb for obj in c[i5][1000:] if obj.isdaughter == i4]
                            x2 = [obj.vb for obj in c[i5][1000:] if obj.isdaughter == i4 and obj.wb > par1['delta']]
                            y1 = [obj.vd for obj in c[i5][1000:] if obj.isdaughter == i4]
                            val1 = scipy.stats.linregress(x1, y1)
                            a[i0, i1, i3, i4, i5, 0] = val1[0]  # slope result
                            a[i0, i1, i3, i4, i5, 1] = len(x2) * 100.0 / len(x1)  # % with low div conc result
                            # % with low birth conc result
                    del c, obs1
                    # print rank, "time taken", time.clock()-tic
                    # print a[i0, i1, i2, i3, 1, 1, :]
                    # print par1['g1_thresh_std'], par1['r_std']
                    # exit()
        print 'I am {0} and I have done one range'.format(rank)
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
        np.save('April17_paper_scripts_8_v1', new_grid)

# V1 has no negative growth whatsoever.
