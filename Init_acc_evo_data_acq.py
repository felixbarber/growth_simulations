#!/usr/bin/env python

import numpy as np
import growth_simulation_accumulator_asymmetric as g
import time
from scipy import stats
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from mpi4py import MPI

td = 1.0
delta = 10.0
# par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01), ('td', td),
#                  ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('CD', 0.58496),
#                  ('g2_std', 0.2), ('d_std', 0.1), ('g1_thresh_std', 0.1)])

par_vals = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 4), ('dt', 0.01), ('td', td),
                 ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('r', 0.5),
                 ('r_std', 0.2), ('d_std', 0.0), ('g1_thresh_std', 0.0)])


# N = 50001
# save_freq = 1000

# N = 10
# save_freq = 2
# num_rep = 8

num_rep = 20
N = 20001
# N = 15
save_freq = 500
num_cells = 3
timer = 0.037  # for 8 cores
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print size
    # print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        # tic = time.clock()
        print N, save_freq, par_vals['modeltype'], num_rep, size, 'changed'
        if par_vals['r'] != 0.5:
            raise ValueError('Not updated')
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [num_rep, N, num_cells, 8]
    a = np.zeros(X)
    if rank == 0:
        print 'Setup matrix:', a.shape
        print 'Expected time taken:', timer * N * 1.0*np.ceil(num_rep*1.0/size*1.0)/3600.0, 'time grown:', par_vals['dt']*par_vals['nstep']*N

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
    tic = time.clock()
    for i0 in xrange(start, stop):  # variable number of repeats for each core
        c = g.discr_gen(par_vals)
        temp = [obj for obj in c if obj.exists]
        for i1 in range(X[1]):
            #
            # tic = time.clock()
            #
            temp0, temp1 = g.starting_popn_seeded_1(temp, par_vals, discr_time=True)
            if np.mod(i1, save_freq) == 0:
                np.save(
                    '../../Documents/init_acc_evo_data/init_acc_evo_data_acq_savedpop_model_{0}_rep_{1}_it_{2}_asymm'.format(str(par_vals['modeltype']),
                                                                                            str(i0), str(i1)), temp1)
            new_c = g.discr_time_1(par_vals, temp0)
            mothers = [obj.mother for obj in new_c[0] if obj.exists]
            del temp  # just to make sure it isn't stored.
            temp = [obj for obj in new_c[0] if obj.exists]
            for i2 in range(num_cells):  # full population included
                if i2 <= 1:  # in this case we deal with mothers and daughters
                    temp2 = np.asarray([obj.vb for obj in temp if obj.isdaughter == i2])
                    temp3 = np.asarray([obj.wb for obj in temp if obj.isdaughter == i2])
                    a[i0, i1, i2, 0] = np.mean(temp2)
                    a[i0, i1, i2, 1] = np.std(temp2)
                    a[i0, i1, i2, 2] = np.mean(temp3)
                    a[i0, i1, i2, 3] = np.std(temp3)
                    a[i0, i1, i2, 4] = np.mean(np.log(temp2))
                    a[i0, i1, i2, 5] = np.std(np.log(temp2))
                    a[i0, i1, i2, 6] = np.mean(np.log(temp3))
                    a[i0, i1, i2, 7] = np.std(np.log(temp3))
                    del temp2, temp3
                else:
                    temp2 = np.asarray([obj.vb for obj in temp])
                    temp3 = np.asarray([obj.wb for obj in temp])
                    a[i0, i1, i2, 0] = np.mean(temp2)
                    a[i0, i1, i2, 1] = np.std(temp2)
                    a[i0, i1, i2, 2] = np.mean(temp3)
                    a[i0, i1, i2, 3] = np.std(temp3)
                    a[i0, i1, i2, 4] = np.mean(np.log(temp2))
                    a[i0, i1, i2, 5] = np.std(np.log(temp2))
                    a[i0, i1, i2, 6] = np.mean(np.log(temp3))
                    a[i0, i1, i2, 7] = np.std(np.log(temp3))
                    del temp2, temp3
            # #
            # print rank, "time taken", time.clock()-tic
            # print a[i0, i1, :, :]
            # exit()
            #
        print 'I am {0} and I have done rep {0}.'.format(str(rank), str(i0)), 'Time taken:', time.clock()-tic
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
        # print 'Time taken =', time.clock()-tic
        # plt.imshow(new_grid)
        # plt.show()
        np.save('./init_acc_evo_data_acq_model_{0}_asymm'.format(par_vals['modeltype']), new_grid)

# 77898.128497s taken total