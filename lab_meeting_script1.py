#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

g2_std = np.linspace(0.0, 0.2, 16)
g1_std = np.linspace(0.0, 0.2, 5)
num_rep = 3
models = [17, 18]
par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.67), ('num_gen', 9), ('K', 1.0/0.67), ('td', 1.0), ('modeltype', 4), ('k_std', 0.2)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.2), ('delta', 1.0)])

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X0, X1, X2, X3 = len(g2_std), len(g1_std), len(models), num_rep
    a = np.zeros((X1, X0, X2, X3, 6, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X0
    for i0 in xrange(start, stop):
        par1['g2_std'] = g2_std[i0]
        for i1 in range(X1):
            par1['g1_thresh_std'] = g1_std[i1]
            for i2 in range(X2):
                par1['modeltype'] = models[i2]
                for i3 in range(X3):
                    obs = g.single_par_meas6(par1)  # note this also returns slopes for full pop
                    a[i1, i0, i2, i3, :, :] = obs[:, :]
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
        np.save('models'+str(models[0])+str(models[1])+'_script1', new_grid)
