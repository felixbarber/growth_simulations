#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

w_frac = np.linspace(0.325, 0.5, 8)
cd = np.linspace(0.45, 0.75, 2)
g1_std = np.linspace(0.0, 0.1, 3)
g2_std = np.linspace(0.0, 0.2, 6)
l_std = np.linspace(0.0, 0.2, 2)
k_std = np.linspace(0.0, 0.2, 2)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 10), ('l_std', 0.0),
             ('g1_delay', 0.0), ('k_std', 0.0), ('w_frac', 0.0)])
models = [15, 16]
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X0, X1, X2, X3, X4, X5, X6 = len(w_frac), len(cd), len(g1_std), len(g2_std), len(l_std), len(k_std), len(models)
    a = np.zeros((X0, X1, X2, X3, X4, X5, X6, 6, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X0
    for i0 in xrange(start, stop):
        par1['w_frac'] = w_frac[i0]
        for i1 in range(X1):
            par1['CD'] = cd[i1]
            for i2 in range(X2):
                par1['g1_thresh_std'] = g1_std[i2]
                for i3 in range(X3):
                    par1['g2_std'] = g2_std[i3]
                    for i4 in range(X4):
                        par1['l_std'] = l_std[i4]
                        for i5 in range(X5):
                            par1['k_std'] = k_std[i5]
                            for i6 in range(X6):
                                par1['modeltype'] = models[i6]
                                obs = g.single_par_meas6(par1)
                                a[i0, i1, i2, i3, i4, i5, i6, :, :] = obs[:6, :]
        print 'I am {0} and I have done one range'.format(rank)
    # Now everyone has made part of the matrix. Send to one processor. Many ways to do this. Broadcast, sendrecv etc
    if rank != 0:
        comm.send(a, dest=0)
    if rank == 0:
        new_grid = np.zeros(np.shape(a))
        new_grid += a
        for p in range(1,size):
            print 'I am 0 and I got from ', p
            new_grid += comm.recv(source=p)
    comm.barrier()
    if rank == 0:
        print 'Time taken =', time.clock()-tic
        # plt.imshow(new_grid)
        # plt.show()
        np.save('model_15_16_K_1', new_grid)

# should take ~ 11 hr
