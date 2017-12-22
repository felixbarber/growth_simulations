#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_accumulator_asymmetric as h
import time
from scipy import stats
import scipy

delta = 15.75

par1 = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 3),
            ('r', 0.68), ('r_std', 0.26), ('delta', delta), ('g1_thresh_std', 0.3)])
c=[]
temp1 = h.discr_gen(par1)
c.append(temp1)
del temp1
# This will initialize the subsequent simulations for this model
temp = h.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
# initial pop seeded from c
temp1, obs1 = h.discr_time_1(par1, temp)

vi = [obj.vi for obj in temp1 if obj.exists]
print np.mean(vi), 2*delta, np.std(vi)