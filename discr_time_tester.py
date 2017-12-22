#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time

# This file tests systematically the deviation between discretized time and discretized generation numbers.

g1_std = np.linspace(0.0, 0.28, 2)
g2_std = np.linspace(0.0, 0.28, 2)
cd = np.linspace(0.5, 1.5, 16)

par1 = dict([('g1_std', 0.0), ('g2_std', 0.0), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0)
            , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0), ('modeltype', 0)])


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X , Y , Z = len(cd) , len(g1_std) , len(g2_std)
    a = np.zeros((X, Y, Z, 6, 2, 3))
    if rank == 0:
        print 'Setup matrix'
    dx = X / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X
    for i in xrange(start, stop):
        par1['CD'] = cd[i]
        fig1 = plt.figure(figsize=[16, 16])
        fig1.suptitle('Mothers $CD =$' + str(np.round(cd[i], 2)))
        fig2 = plt.figure(figsize=[16, 16])
        fig2.suptitle('Daughters $CD =$' + str(np.round(cd[i], 2)))
        for j in range(Y):
            par1['g1_thresh_std'] = g1_std[j]
            for k in range(Z):
                par1['g2_std'] = g2_std[k]

                obs1, obs2, tg1, tg2 = g.single_par_meas5(par1)  # Obs1 are leaf cells, obs2 are entire tree cells
                obs3, tg3 = g.single_par_meas4(par1)
                a[i, j, k, :, :, 0] = obs1[:6, :]  # save data for the different kinds of measurements we are making
                a[i, j, k, :, :, 1] = obs2[:6, :]
                a[i, j, k, :, :, 2] = obs3[:6, :]
                np.save('./data/tgrow_m_model0_discr_time_leaf_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg1[0])
                np.save('./data/tgrow_m_model0_discr_time_tree_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg2[0])
                np.save('./data/tgrow_m_model0_discr_genr_tree_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg3[0])
                np.save('./data/tgrow_d_model0_discr_time_leaf_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg1[1])
                np.save('./data/tgrow_d_model0_discr_time_tree_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg2[1])
                np.save('./data/tgrow_d_model0_discr_genr_tree_cd_' + str(i) + '_s1_' + str(j) + '_s2_' + str(k), tg3[1])
                ax1 = fig1.add_subplot(len(g1_std), len(g2_std), k + 1 + j * (len(g2_std)))
                sns.distplot(tg1[0], label='discr time leaf')
                sns.distplot(tg2[0], label='discr time tree')
                sns.distplot(tg3[0], label='discr gen tree')
                plt.legend()
                plt.title('$\sigma_{i}=$'+str(np.round(g1_std[j], 2))+' $\sigma_{G2}=$'+str(np.round(g2_std[k], 2)))
                if j == len(g1_std)-1:
                    plt.xlabel('Growth time $t/t_d$')
                ax2 = fig2.add_subplot(len(g1_std), len(g2_std), k + 1 + j * (len(g2_std)))
                sns.distplot(tg1[1], label='discr time leaf')
                sns.distplot(tg2[1], label='discr time tree')
                sns.distplot(tg3[1], label='discr gen tree')
                plt.legend()
                plt.title('$\sigma_{i}=$' + str(np.round(g1_std[j], 2)) + ' $\sigma_{G2}=$' + str(np.round(g2_std[k], 2)))
                if j == len(g1_std) - 1:
                    plt.xlabel('Growth time $t/t_d$')
        fig1.savefig('tgrow_m_model0_cd_' + str(i) + '.eps', dpi=fig1.dpi)
        fig2.savefig('tgrow_d_model0d_cd_' + str(i) + '.eps', dpi=fig2.dpi)
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
        np.save('whi5_noiseless_adder_discrgen_tree_cd_75_K_1', new_grid)
