#!/usr/bin/env python

import numpy as np
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
from mpi4py import MPI

# This script compiles all the different variants of the dilution model, and calculates the robustness to the type of
# noise which is appropriate for each variant. Note that we should have equal distribution of Whi5 for each of the two
# progeny. We have also neglected to include noise in the division ratio.

delta, td = 10.0, 1.0
par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', None), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('K', delta/td),
             ('CD', 1.0), ('r', 1.0), ('r_std', 0.0), ('g2_std', 0.0), ('d_std', None), ('k_std', None),
                 ('g1_thresh_std', None)])
# Model list:
models_descr = ['Noisy synthesis rate', 'Noisy synthesis rate NNG', 'Noisy integrator', 'Noisy integrator NNG',
                'Fixed r, noisy integrator', 'Fixed r, noisy integrator NNG']
# note that without noise in division ratio, these should
# map exactly to each other under the rescaling delta = k*td. Should therefore keep these factors the same in each.

L = 7
models = [5, 10, 17, 18, 23, 24]
g1_thresh_std = np.linspace(0.0, 0.3, L)
k_std = np.linspace(0.0, 0.3, L)
d_std = np.linspace(0.0, 0.3, L)
vals = [['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'k_std'], ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std'],
        ['g1_thresh_std', 'd_std'], ['g1_thresh_std', 'd_std']]
pars = [[g1_thresh_std, k_std], [g1_thresh_std, k_std], [g1_thresh_std, d_std], [g1_thresh_std, d_std],
        [g1_thresh_std, d_std], [g1_thresh_std, d_std]]
num_rep = 20  # number of repeats for each condition
num_celltype = 2
num_sims = 2  # number of different simulations to be run
num_meas = 3  # number of different kinds of measurements

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [len(models), L, L, num_rep, num_celltype, num_sims, num_meas]
    a = np.zeros(X)
    if rank == 0:
        print 'Setup matrix'
    dx = X[0] / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X[0]
    for i0 in xrange(start, stop):  # varying the different growth conditions
        par1 = par_vals
        par1['modeltype'] = models[i0]
        for i1 in range(X[1]):
            par1[vals[i0][0]] = pars[i0][0][i1]
            for i2 in range(X[2]):
                par1[vals[i0][1]] = pars[i0][1][i2]
                for i3 in range(X[3]):
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
                    for i4 in range(X[4]):  # celltype
                        for i5 in range(X[5]):  # sim number
                            x1 = [obj.vb for obj in c[i5][1000:] if obj.isdaughter == i4]
                            x2 = [obj.vb for obj in c[i5][1000:] if obj.isdaughter == i4 and obj.wd / obj.vd < 1.0]
                            x3 = [obj.vb for obj in c[i5][1000:] if obj.isdaughter == i4 and obj.wb / obj.vb < 1.0]
                            y1 = [obj.vd for obj in c[i5][1000:] if obj.isdaughter == i4]
                            val1 = scipy.stats.linregress(x1, y1)
                            a[i0, i1, i2, i3, i4, i5, 0] = val1[0]  # slope result
                            a[i0, i1, i2, i3, i4, i5, 1] = len(x2) * 100.0 / len(x1)  # % with low div conc result
                            a[i0, i1, i2, i3, i4, i5, 2] = len(x3) * 100.0 / len(x1)
                            # % with low birth conc result
                    del c, obs1
                    # print rank, "time taken", time.clock()-tic
                    # print a[i0, i1, i2, i3, 1, 0, :]
                    # print par1['g1_thresh_std'], par1['d_std'], par1['k_std']
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
        np.save('March17_dilution_symmetric_V1', new_grid)

# V1 has no negative growth in any of models 10, 18, 24. Took 6235.861612s
