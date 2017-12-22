#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

cd = np.linspace(0.4, 0.8, 8)
g1_std = np.linspace(0.0, 0.2, 3)
g2_std = np.linspace(0.0, 0.2, 5)
l_std = np.linspace(0.0, 0.25, 6)
d_std = np.linspace(0.0, 0.2, 5)
models = [17, 18]

par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('CD', 0.67), ('num_gen', 9), ('td', 1.0), ('modeltype', 4)
             , ('l_std', 0.2), ('g1_delay', 0.0), ('d_std', 0.2), ('K', 100.0/0.67), ('delta', 100.0)])

vals = ['CD', 'g1_thresh_std', 'g2_std', 'l_std', 'd_std', 'modeltype']
pars = [cd, g1_std, g2_std, l_std, d_std, models]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X0, X1, X2, X3, X4, X5 = len(cd), len(g1_std), len(g2_std), len(l_std), len(d_std), len(models)
    a = np.zeros((X0, X1, X2, X3, X4, X5, 6, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X0
    for i0 in xrange(start, stop):
        par1[vals[0]] = pars[0][i0]
        for i1 in range(X1):
            par1[vals[1]] = pars[1][i1]
            for i2 in range(X2):
                par1[vals[2]] = pars[2][i2]
                for i3 in range(X3):
                    par1[vals[3]] = pars[3][i3]
                    for i4 in range(X4):
                        par1[vals[4]] = pars[4][i4]
                        for i5 in range(X5):
                            par1[vals[5]] = pars[5][i5]
                            obs = g.single_par_meas6(par1)  # note this also returns slopes for full pop
                            a[i0, i1, i2, i3, i4, i5, :, :] = obs[:, :]
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
        np.save('models'+str(models[0])+str(models[1])+'_script2', new_grid)
