#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

cd = np.linspace(0.5, 0.9, 8)
g1_std = np.linspace(0.0, 0.2, 3)
g2_std = np.linspace(0.0, 0.2, 3)
l_std = np.linspace(0.0, 0.2, 3)
k_vals = np.linspace(0.5, 1.5, 3)
g1_t_std = np.linspace(0.0, 0.2, 5)
g1_del = np.linspace(0.0, 0.2, 5)


par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 9),
             ('g1_delay', 0.1), ('l_std', 0.0)])

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X0, X1, X2, X3, X4, X5, X6 = len(cd), len(g1_std), len(g2_std), len(l_std), len(k_vals), len(g1_t_std), len(g1_del)
    a = np.zeros((X0, X1, X2, X3, X4, X5, X6, 6, 3, 2))
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X1
    for i0 in xrange(start, stop):
        par1['CD'] = cd[i0]
        for i1 in range(X1):
            par1['g1_thresh_std'] = g1_std[i1]
            for i2 in range(X2):
                par1['g2_std'] = g2_std[i2]
                for i3 in range(X3):
                    par1['l_std'] = l_std[i3]
                    for i4 in range(X4):
                        par1['K'] = k_vals[i4]
                        for i5 in range(X5):
                            par1['g1_std'] = g1_t_std[i5]
                            for i6 in range(X6):
                                par1['g1_del'] = g1_del[i6]
                                par1['modeltype'] = 9  # minimum in Vb
                                obs3, tg3 = g.single_par_meas6(par1)  # note this also returns slopes for full pop
                                a[i0, i1, i2, i3, i4, i5, i6, :, :, 0] = obs3[:6, :]
                                par1['modeltype'] = 4  # no minimum in Vb
                                obs3, tg3 = g.single_par_meas6(par1)  # note this also returns slopes for full pop
                                a[i0, i1, i2, i3, i4, i5, i6, :, :, 1] = obs3[:6, :]
                                # basepath = '/home/felix/simulation_data/discr_time_tester1_data/'
                                # np.save(basepath + 'tgrow_m_model'+str(par1['modeltype'])+'_discr_genr_tree_cd_'
                                #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k) + '_sl_' + str(h) + '_k_' + str(u), tg3[0])
                                # np.save(basepath + 'tgrow_d_model'+str(par1['modeltype'])+'_discr_genr_tree_cd_'
                                #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k) + '_sl_' + str(h) + '_k_' + str(u), tg3[1])
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
        np.save('model_9_4_g1_delay', new_grid)
