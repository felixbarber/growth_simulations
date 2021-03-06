#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

cd = np.linspace(0.4, 1.0, 25)
g2_std = np.linspace(0.01, 0.25, 25)


par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('CD', 0.67), ('num_gen', 9), ('td', 1.0), ('modeltype', 4)
             , ('l_std', 0.0), ('g1_delay', 0.0), ('d_std', 0.2), ('K', 100.0/0.67), ('delta', 100.0)])

models = [17, 18, 4, 9, 5, 10]

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
    num_reps = 3
    X0, X1, X2, X3 = len(cd), len(g2_std), len(models), num_reps
    a = np.zeros((X0, X1, X2, X3, 2, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X0
    for i0 in xrange(start, stop):
        par1['CD'] = cd[i0]
        for i1 in range(X1):
            par1['g2_std'] = g2_std[i1]
            for i2 in range(num_reps):
                c = g.discr_gen(par1)  # note this also returns slopes for full pop
                for i3 in range(3):  # mothers, daughters and population
                    if i3 < 2:
                        x1 = [obj.vb for obj in c[1000:] if obj.isdaughter == i3 and obj.wd / obj.vd < 1.0]
                        y1 = [obj.vd for obj in c[1000:] if obj.isdaughter == i3 and obj.wd / obj.vd < 1.0]
                        x = [obj.vb for obj in c[1000:] if obj.isdaughter == i3]
                    else:
                        x1 = [obj.vb for obj in c[1000:] if obj.wd / obj.vd < 1.0]
                        y1 = [obj.vd for obj in c[1000:] if obj.wd / obj.vd < 1.0]
                        x = [obj.vb for obj in c[1000:]]
                    a[i0, i1, i2, i3] = len(x1)*100.0/len(x)
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
        np.save('model_4_finegrid_lowconc', new_grid)
