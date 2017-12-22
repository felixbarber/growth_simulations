#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy


# This script will assess the effect of biologically relevant variability around the observed values for each growth
# condition. Some variables are initialized based on observations of asymmetry ratio, G1 length difference etc.

# Simulations will be initialized by randomly selecting leaf cells from a distribution which is chosen to have converged
# to the final distribution. This is to be done by using a discretized generation simulation, run for 10 generations.
# The leaf cells of this population are used to seed 5 parallel simulations, both discretized time and discretized gen.
# We then look at the mean slope for each parameter value, and check the result

data = np.load('save_data_2.npy')

r = 0.52
delta = 10.0

par_set = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
                ('num_gen', 10), ('modeltype', 16)])

model_descr = ['Noisy synth const frac no neg growth']

num_med = 5
num_rep = 5
len_var = 3  # number of sampling points for each variable
len_unfx = 4
g1_std = np.linspace(0.0, 0.1, 3)  # as a fraction of KCD
k_std = np.linspace(0.0, 0.5, 6)  # as a fraction of K
# k_std = np.array([0.5])
num_celltype = 2
num_stds = 2  # number of standard errors on either side of the inferred value to be searched
num_sims = 3  # number of different kinds of data stored: discr gen, discr time, theory
print "G1_std:", g1_std
print "k_std:", k_std

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [num_med, len_var, len(g1_std), len_var, len_var, len(k_std), len_var, num_rep, num_celltype, num_sims]
    a = np.zeros(X)
    if rank == 0:
        print 'Setup matrix'
    dx = X[0] / size
    start = dx * rank
    stop = start + dx
    if rank == size - 1:
        stop = X[0]
    for i0 in xrange(start, stop):  # varying the different growth conditions
        # Reset par1 for each new growth condition so we start with a blank slate
        par1 = par_set

        # Calculating growth condition-specific variables
        temp1 = data[i0, 28, :]  # noise in gr
        # temp2 = data[i0, 29, :]  # noise in bud length -- gr version
        temp2 = data[i0, 27, :]  # noise in bud length -- td version
        # temp3 = data[i0, 30, :]  # mean bud length -- gr version
        temp3 = data[i0, 13, :]  # mean bud length -- td version
        temp4 = data[i0, 35, :]  # Whi5 daughter fraction f.
        l_std = np.linspace(temp1[0] - num_stds * temp1[1], temp1[0] + num_stds * temp1[1], len_var)
        g2_std = np.linspace(temp2[0] - num_stds*temp2[1], temp2[0] + num_stds*temp2[1], len_var)
        cd = np.linspace(temp3[0] - num_stds*temp3[1], temp3[0] + num_stds*temp3[1], len_var)
        f = np.linspace(temp4[0] - num_stds*temp4[1], temp4[0] + num_stds*temp4[1], len_var)
        # Gives a * Hopefully * generous estimate of the variability in Whi5 daughter fraction

        print "Growth medium ", i0, "l_std", temp1[0]
        print "Growth medium ", i0, "G2 std deviation", temp2[0]
        print "Growth medium ", i0, "CD length", temp3[0]
        print "Growth medium ", i0, "w_frac", temp4[0]
        # gives a * hopefully * generous range of observable variability
        pars = ['placeholder', cd, g1_std, g2_std, l_std, k_std, f]
        vals = ['placeholder', 'CD', 'g1_thresh_std', 'g2_std', 'l_std', 'k_std', 'w_frac']
        for i1 in range(X[1]):
            par1[vals[1]] = pars[1][i1]
            par1['K'] = delta / par1['CD']  # Keep the value of Delta fixed throughout
            for i2 in range(X[2]):
                par1[vals[2]] = pars[2][i2]
                for i3 in range(X[3]):
                    par1[vals[3]] = pars[3][i3]
                    for i4 in range(X[4]):
                        par1[vals[4]] = pars[4][i4]
                        for i5 in range(X[5]):
                            par1[vals[5]] = pars[5][i5]
                            for i6 in range(X[6]):
                                par1[vals[6]] = pars[6][i6]
                                for i7 in range(X[7]):
                                    # tic = time.clock()
                                    c = g.discr_gen(par1)
                                    # This will initialize the subsequent simulations for this model
                                    temp = g.starting_popn_seeded([obj for obj in c if obj.exists], par1)
                                    # initial pop seeded from c
                                    c1, obs1 = g.discr_time_1(par1, temp)
                                    for i8 in range(X[8]):
                                        x1 = [obj.vb for obj in c if obj.isdaughter == i8]
                                        y1 = [obj.vd for obj in c if obj.isdaughter == i8]
                                        x2 = [obj.vb for obj in c1 if obj.isdaughter == i8]
                                        y2 = [obj.vd for obj in c1 if obj.isdaughter == i8]
                                        val1 = scipy.stats.linregress(x1, y1)
                                        val2 = scipy.stats.linregress(x2, y2)
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0] = val1[0]  # discrete gen sim result
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1] = val2[0]  # discrete time sim result
                                    del c, c1, obs1, temp
                                    a[i0, i1, i2, i3, i4, i5, i6, i7, 0, 2] = g.slope_vbvd_m(par1,
                                                                                                par1['g1_thresh_std'] *
                                                                                                par1['K'] * par1['CD'],
                                                                                                par1['g2_std'] * par1[
                                                                                                    'td'])
                                    # theory mothers
                                    a[i0, i1, i2, i3, i4, i5, i6, i7, 1, 2] = g.slope_vbvd_func(par1,
                                                                                                par1['g1_thresh_std'] *
                                                                                                par1['K'] * par1['CD'],
                                                                                                par1['g2_std'] * par1[
                                                                                                    'td'])
                                    # theory daughters
                                    # print rank, "time taken", time.clock()-tic
                                    # print a[i0, i1, i2, i3, i4, i5, i6, i7, 1, :]
                                    # exit()
            print 'I am {0} and I have done one range'.format(rank)
        del cd, g2_std, l_std, f  # delete the new variables
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
        np.save('Feb17_test_models3_model'+str(par1['modeltype'])+'_V3', new_grid)

# should take ~ 15hr as is. Started 3pm. took 18 hrs.
