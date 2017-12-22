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
# The leaf cells of this population are used to seed 50 parallel simulations, both discretized time and discretized gen.
# We then look at the mean and standard deviation of the resultant observed slopes and test their deviation from the
# theoretically predicted slopes.

data = np.load('growth_condition_data.npy')

r = 0.52
delta = 10.0

par_set = dict([('g1_std', 0.0), ('g2_std', 0.2), ('g1_thresh_std', 0.2), ('dt', 0.01),
                ('td', 1.0), ('g1_delay', 0.0), ('l_std', 0.2), ('num_s1', 500), ('nstep', 500), ('num_gen', 10),
                ('modeltype', 16)])

models = [15, 16]
model_descr = ['Noisy synth const frac', 'Noisy synth const frac no neg growth']

num_med = 5
num_rep = 5
g1_std = np.linspace(0.0, 0.2, 3)  # as a fraction of KCD
k_std = np.linspace(0.0, 0.2, 3)  # as a fraction of K
len_var = 3
num_celltype = 2
num_sim = 2
frac_var = 0.05
frac_var_f = 0.03
len_cd = 5
num_obs = 3

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print "I am processor {0} and I am hungry!".format(rank)
    if rank == 0:
        tic = time.clock()
        # print 'Notice that they are out of order, and if you run it over and over the order changes.'
    # KISS first. Make everyone make a matrix
    X = [num_med, len_cd, len(g1_std), len_var, len_var, len(k_std), len_var, num_rep, num_celltype, num_sim, num_obs]
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
        v1 = data[2, i0, 2]  # asymmetry ratio
        v2 = (data[0, i0, 10]-data[1, i0, 10]) * data[2, i0, 17]  # (<tg1>d-<tg1>m)* <GR>
        # This expression was derived assuming no noise in GR
        v3 = data[2, i0, 12] / data[2, i0, 8]  # Time spent in budded phase <tbud>/<td>
        v4 = v1 * np.exp(v2) / (1 + v1 * np.exp(v2))  # fraction of Whi5 given to daughter cells
        v5 = data[2, i0, 18]  # Growth rate CV
        v6 = data[2, i0, 19]  # standard deviation in budded phase duration scaled by average doubling time.
        v7 = data[2, i0, 30] / data[2, i0, 8]  # Standard error of the mean in budded phase timing /<td>
        v8 = data[2, i0, 31] * v6
        # standard error in the std deviation statistic for CD timing * std deviation statistic
        v9 = data[2, i0, 31] * v5
        # standard error in the std deviation statistic for growth rate * std deviation statistic
        v10 = data[2, i0, 32] / data[2, i0, 8]  # standard error of the mean in doubling time / <td>.
        v11 = data[2, i0, 33] / data[2, i0, 17]  # standard error of the mean in growth rate / <lambda>.
        # Defining estimates of variability in all the biologically constrained parameters.
        # temp=(v7 + v3 * v10)
        cd = np.linspace(v3 - 2 * (v7 + v3 * v10), v3 + 2 * (v7 + v3 * v10), len_cd)
        g2_std = np.linspace(v6 - 3 * (v8 + v10 * v6), v6 + 3 * (v8 + v10 * v6), len_var)
        l_std = np.linspace(v5 - 3 * (v9 + v11 * v5), v5 + 3 * (v9 + v11 * v5), len_var)
        # Gives a * Hopefully * generous estimate of the variability in Whi5 daughter fraction
        f = np.linspace(v4-frac_var_f, v4+frac_var_f, len_var)
        print "Growth medium ", i0, "G2 std deviation", g2_std
        print "Growth medium ", i0, "CD length", cd
        print "Growth medium ", i0, "l_std", l_std
        print "Growth medium ", i0, "w_frac", f
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
                                    c = g.discr_gen(par1)
                                    # This will initialize the subsequent simulations for this model
                                    temp = g.starting_popn_seeded([obj for obj in c if obj.exists], par1)
                                    # initial pop seeded from c
                                    c1, obs1 = g.discr_time_1(par1, temp)
                                    for i8 in range(X[8]):
                                        x1 = [obj.vb for obj in c if obj.isdaughter == i8]
                                        y1 = [obj.vd for obj in c if obj.isdaughter == i8]
                                        z1 = np.mean([obj.t_grow - obj.t_bud for obj in c if obj.isdaughter == i8])
                                        u1 = [obj for obj in c if obj.mother and obj.mother.isdaughter == i8 and obj.wb < obj.vb]
                                        uu1 = []
                                        for obj in u1:  # this step is necessary just to avoid duplicates
                                            if obj.mother in uu1:
                                                continue
                                            uu1.append(obj.mother)
                                        # print len(uu1), len(u1)
                                        w1 = len(uu1) * 100.0 / len(x1)
                                        # print len(u1) * 100.0 / len(x1)
                                        # print len([obj for obj in c if obj.wb<obj.vb and obj.isdaughter])*100.0/len([obj for obj in c if obj.isdaughter])
                                        # print len(
                                        #     [obj for obj in c if obj.wb < obj.vb and obj.isdaughter == 0]) * 100.0 / len(
                                        #     x1)
                                        val1 = scipy.stats.linregress(x1, y1)
                                        del x1, y1, u1, uu1
                                        x2 = [obj.vb for obj in c1 if obj.isdaughter == i8]
                                        y2 = [obj.vd for obj in c1 if obj.isdaughter == i8]
                                        val2 = scipy.stats.linregress(x2, y2)
                                        z2 = np.mean([obj.t_grow - obj.t_bud for obj in c1 if obj.isdaughter == i8])
                                        u2 = [obj for obj in c1 if obj.mother and obj.mother.isdaughter == i8 and obj.wb < obj.vb]
                                        uu2 = []
                                        for obj in u2:  # this step is necessary just to avoid duplicates
                                            if obj.mother in uu2:
                                                continue
                                            uu2.append(obj.mother)
                                        w2 = len(uu2) * 100.0 / len(x2)
                                        # print w1, w2, z1, z2
                                        del x2, y2, u2, uu2
                                        # exit()
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0, 0] = val1[0]  # discrete gen sim result Slope
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1, 0] = val2[0]  # discrete time sim result Slope
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0, 1] = z1  # discrete gen sim result average G1 len
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1, 1] = z2  # discrete time sim result G1 len
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0, 2] = w1  # discrete gen sim result percent low conc cells
                                        a[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1, 2] = w2  # discrete time sim result percent low conc cells
                                    del c, c1, obs1, temp
                                    # a[i0, i1, i2, i3, i4, i5, i6, i7, 0, 2, 0] = g.slope_vbvd_m(par1,
                                    #                                                             par1['g1_thresh_std'] *
                                    #                                                             par1['K'] * par1['CD'],
                                    #                                                             par1['g2_std'] * par1[
                                    #                                                                 'td'])
                                    # # theory mothers
                                    # a[i0, i1, i2, i3, i4, i5, i6, i7, 1, 2, 0] = g.slope_vbvd_func(par1,
                                    #                                                             par1['g1_thresh_std'] *
                                    #                                                             par1['K'] * par1['CD'],
                                    #                                                             par1['g2_std'] * par1[
                                    #                                                                 'td'])
                                    # theory daughters
                                    # print a[i0, i1, i2, i3, i4, i5, i6, i7, 0, 2], a[i0, i1, i2, i3, i4, i5, i6, i7, 0, 0], a[i0, i1, i2, i3, i4, i5, i6, i7, 0, 1]
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
        np.save('Feb17_test_models3_model'+str(par1['modeltype'])+'_1', new_grid)
