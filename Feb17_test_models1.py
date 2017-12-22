#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

# This script will consider the effect of initialization on the simulations.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 50 parallel simulations, both discretized time and discretized gen.
# We then look at the mean and standard deviation of the resultant observed slopes and test their deviation from the
# theoretically predicted slopes.

r = 0.52
x = 0.2
f = r*2**x/(1+r*2**x)
cd = np.log(1+r)/np.log(2)
delta = 10.0

par_set = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('dt', 0.01), ('CD', cd), ('K', delta/cd),
                ('td', 1.0), ('g1_delay', 0.0), ('l_std', 0.2), ('d_std', 0.0), ('delta', delta)])

# These parameter settings are defined anew for each different model.
par1_vals = [[['modeltype', 17], ['d_std', 0.2], ['num_gen', 10]], [['modeltype', 17], ['d_std', 0.0], ['num_gen', 10]],
             [['modeltype', 5], ['k_std', 0.2], ['num_gen', 10]], [['modeltype', 15], ['w_frac', f], ['k_std', 0.2],
                                                                   ['num_gen', 10]]]
models = [17, 17, 5, 15]
model_descr = ['Noisy integrator', 'Noiseless integrator', 'Noisy synth', 'Noisy synth const frac']

num_s1 = [100, 200, 300, 400, 500, 600, 700]
num_gen1 = [3, 4, 5, 6, 7]
num_rep = 100
num_celltype = 2

vals = ['modeltype',  'num_s1', 'num_gen1']
pars = [models, num_s1, num_gen1]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X0, X1, X2, X3, X4 = len(models), len(num_s1), len(num_gen1), num_rep, num_celltype
    a = np.zeros([X0, X1, X2, X3, X4, 2])
    if rank == 0:
        print 'Setup matrix'
    dx = X0 / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X0
    for i0 in xrange(start, stop):
        # Reset par1 for each new model, with settings appropriate for that model
        par1 = par_set
        for temp in range(len(par1_vals[i0])):
            par1[par1_vals[i0][temp][0]] = par1_vals[i0][temp][1]
        c = g.discr_gen(par1)  # This will initialize the subsequent simulations for this model
        for i1 in range(X1):
            par1[vals[1]] = pars[1][i1]
            for i2 in range(X2):
                par1[vals[2]] = pars[2][i2]
                par1['nstep'] = int(pars[2][i2] * 1.0 / par1['dt'])
                for i3 in range(X3):
                    temp = g.starting_popn_seeded([obj for obj in c if obj.exists], par1)  # initial pop seeded from c
                    c1, obs1 = g.discr_time_1(par1, temp)
                    del temp
                    temp = g.starting_popn_seeded([obj for obj in c if obj.exists], par1)  # initial pop seeded from c
                    c2 = g.discr_gen_1(par1, temp)
                    del temp
                    for i4 in range(X4):
                        x1 = [obj.vb for obj in c1 if obj.isdaughter == i4]
                        y1 = [obj.vd for obj in c1 if obj.isdaughter == i4]
                        x2 = [obj.vb for obj in c2 if obj.isdaughter == i4]
                        y2 = [obj.vd for obj in c2 if obj.isdaughter == i4]
                        val1 = scipy.stats.linregress(x1, y1)
                        val2 = scipy.stats.linregress(x2, y2)
                        a[i0, i1, i2, i3, i4, 0] = val1[0]
                        a[i0, i1, i2, i3, i4, 1] = val2[0]
            print 'I am {0} and I have done one range'.format(rank)
        del c
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
        np.save('feb17_test_models', new_grid)
