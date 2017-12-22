#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time


# This file differs only from parallel_var in that it uses discretized generations rather than discretized time (comes
# in with using g.single_par_meas2 to produce a population of cells. Output is identical but saved differently.
g1_std = np.linspace(0.0, 0.62, 32)
#g1_std = a[::-1]

g2_std = np.linspace(0.0, 0.62, 32)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.05), ('g1_thresh_std', 0.05), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0)])

if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    #KISS first. Make everyone make a matrix
    X , Y = len(g1_std) , len(g2_std)
    a = np.zeros((X, Y, 6, 2))
    if rank == 0:
        print 'Setup matrix'
    dx = X / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X
    for i in xrange(start, stop):
        par1['g1_thresh_std'] = g1_std[i]
        for j in range(Y):
            par1['g2_std'] = g2_std[j]
            a[i, j, :, :] = g.single_par_meas2(par1)
            np.save('hi_'+str(i)+str(j), a[i,j,:,:])
        print 'I am {0} and I have done one range'.format(rank)
    ##Now everyone has made part of the matrix. Send to one processor. Many ways to do this. Broadcast, sendrecv etc
    if rank != 0:
        comm.send(a, dest=0)
    if rank == 0:
        new_grid = np.zeros(np.shape(a))
        new_grid += a
        for p in range(1,size):
            print 'I am 0 and I got from ',p
            new_grid += comm.recv(source = p)
    comm.barrier()
    if rank == 0:
        print 'Time taken =',time.clock()-tic
        #plt.imshow(new_grid)
        #plt.show()
        np.save('whi5_noiseless_adder_discrgen_tree_cd_75_K_1', new_grid)
