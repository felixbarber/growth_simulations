#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

g1_std = np.linspace(0.0, 0.25, 6)
g2_std = np.linspace(0.0, 0.25, 6)
l_std = np.linspace(0.0, 0.2, 2)
cd = np.linspace(0.5, 1.0, 8)
k_std = np.linspace(0.0, 0.2, 2)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 10), ('K', 1.0), ('td', 1.0), ('modeltype', 10), ('l_std', 0.0)])

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X, Y, Z, W, V = len(cd), len(g1_std), len(g2_std), len(l_std), len(k_std)
    a = np.zeros((X, Y, Z, 6, 2, W, V))
    #a = np.zeros((X, Y, Z, 6, 2, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X
    for i in xrange(start, stop):
        par1['CD'] = cd[i]
        for j in range(Y):
            par1['g1_thresh_std'] = g1_std[j]
            for k in range(Z):
                par1['g2_std'] = g2_std[k]
                for h in range(W):
                    par1['l_std'] = l_std[h]
                    for f in range(V):
                        par1['k_std'] = k_std[f]
                        # obs1, obs2, tg1, tg2 = g.single_par_meas5(par1)  # Obs1 are leaf cells, obs2 are entire tree cells
                        obs3, tg3 = g.single_par_meas4(par1)
                        a[i, j, k, :, :, h, f] = obs3[:6, :]
                        basepath = '/home/felix/simulation_data/discr_time_tester1_data/'
                        # np.save(basepath + 'tgrow_m_model'+str(par1['modeltype'])+'_discr_time_leaf_cd_'
                        #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg1[0])
                        # np.save(basepath + 'tgrow_m_model'+str(par1['modeltype'])+'_discr_time_tree_cd_'
                        #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg2[0])
                        np.save(basepath + 'tgrow_m_model'+str(par1['modeltype'])+'_discr_genr_tree_cd_'
                                + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg3[0])
                        # np.save(basepath + 'tgrow_d_model'+str(par1['modeltype'])+'_discr_time_leaf_cd_'
                        #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg1[1])
                        # np.save(basepath + 'tgrow_d_model'+str(par1['modeltype'])+'_discr_time_tree_cd_'
                        #         + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg2[1])
                        np.save(basepath + 'tgrow_d_model'+str(par1['modeltype'])+'_discr_genr_tree_cd_'
                                + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg3[1])
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
        np.save('discr_time_tester1_model'+str(par1['modeltype'])+'_K_1', new_grid)
