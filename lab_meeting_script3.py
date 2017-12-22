#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
import scipy
from scipy import stats

cd = np.linspace(0.4, 1.0, 16)
g2_std = np.linspace(0.01, 0.25, 16)


par1 = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.0), ('nstep', 900), ('dt', 0.01), ('t_delay', 0.0)
            , ('CD', 0.67), ('num_gen', 9), ('td', 1.0), ('modeltype', 4)
             , ('l_std', 0.0), ('g1_delay', 0.0), ('d_std', 0.2), ('K', 100.0/0.67), ('delta', 100.0), ('k_std', 0.0)])

models = [17, 18, 4, 9, 5, 10]

vals = ['CD', 'g2_std', 'modeltype']
pars = [cd, g2_std, models]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    num_reps = 1
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
        par1[vals[0]] = pars[0][i0]
        for i1 in range(X1):
            par1[vals[1]] = pars[1][i1]
            for i2 in range(X2):
                par1[vals[2]] = pars[2][i2]
                for i3 in range(X3):
                    c = g.discr_gen(par1)  # note this also returns slopes for full pop
                    for i4 in range(3):  # mothers, daughters and population
                        if i4 < 2:
                            x1 = [obj.vb for obj in c[1000:] if obj.isdaughter == i4 and obj.wd / obj.vd < 1.0]
                            x = [obj.vb for obj in c[1000:] if obj.isdaughter == i4]
                            y = [obj.vd for obj in c[1000:] if obj.isdaughter == i4]
                        else:
                            x1 = [obj.vb for obj in c[1000:] if obj.wd / obj.vd < 1.0]
                            x = [obj.vb for obj in c[1000:]]
                            y = [obj.vd for obj in c[1000:]]
                        a[i0, i1, i2, i3, 0, i4] = len(x1)*100.0/len(x)
                        temp = scipy.stats.linregress(x, y)
                        a[i0, i1, i2, i3, 1, i4] = temp[0]
                        # print a[i0, i1, i2, i3, :, i4]
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
        np.save('./lab_meeting_figures/modelcomp_finegrid_g1_0', new_grid)
