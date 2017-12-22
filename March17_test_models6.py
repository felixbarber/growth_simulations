#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy

# The purpose of this script is to test whether our predictions of the behaviour of models 23 and 24 are accurate, and
# whether there is really such a weak dependence on noise in passage through Start in this model.


def wbwb(par1, celltype):  # units should be volume and time for s_i and s_cd respectively.
    if par1['modeltype'] in [23, 24]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 2:
            val = np.mean((temp**2+1)/(1+temp)**2)*(3+par1['d_std']**2)*par1['delta']**2/(2-np.mean((temp**2+1)/(temp+1)**2))
    return val


def vbvb(par1, celltype):
    if par1['modeltype'] in [23, 24]:
        # if celltype == 0:  # mothers
        #     val = vivi(par1, celltype=2)
        if celltype == 1:  # daughters
            val = par1['r']**2*(1+par1['r_std']**2)*(wbwb(par1, celltype=2)+par1['delta']**2*par1['g1_thresh_std']**2)
        # if celltype == 2:  # population
        #     val = 0.5*vivi(par1, celltype=2)*(par1['r']**2*(1+par1['r_std']**2)+1)
    return val


def vdvb(par1, celltype):
    if par1['modeltype'] in [23, 24]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        # if celltype == 0:
        #     val = (2*par1['delta']**2+vivi(par1, celltype=2)*np.mean(1.0/(1+temp))) * (1+par1['r'])
        if celltype == 1:
            val = (1+par1['r'])*(wbwb(par1, celltype=2)+par1['delta']**2)*np.mean(temp**2/(1+temp))
        # elif celltype == 2:
        #     val = 0.5*((2*par1['delta']**2+vivi(par1, celltype=2)*np.mean(1.0/(1+temp))) * (1+par1['r'])+(1+par1['r'])*
        #                (2*par1['delta']**2*par1['r']+vivi(par1, celltype=2)*np.mean(temp**2/(1+temp))))
    return val

def vb(par1, celltype):
    if par1['modeltype'] in [23, 24]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 1:
            val = par1['r']*par1['delta']
    return val

def vd(par1, celltype):
    if par1['modeltype'] in [23, 24]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 1:
            val = (1+par1['r'])*2*par1['delta']*np.mean(temp/(1+temp))
    return val

def wb(par1, celltype):
    if par1['modeltype'] in [23, 24]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 1:
            val = 2*par1['delta']*np.mean(temp/(1+temp))
    return val

delta = 10.0
par1 = dict([('g1_std', 0.0), ('dt', 0.01), ('td', 1.0), ('g1_delay', 0.0), ('num_s1', 500), ('nstep', 500),
                ('num_gen', 9), ('modeltype', 24), ('delta', delta), ('d_std', 0.0), ('g1_thresh_std', 0.0), ('r', 0.68),
             ('r_std', 0.2)])
def slope_vbvd(par1, celltype):
    val = (vdvb(par1, celltype) - vd(par1, celltype) * vb(par1, celltype)) /\
        (vbvb(par1, celltype) - vb(par1, celltype) ** 2)
    return val

c = g.discr_gen(par1)
i0=1
x=np.asarray([obj.vb for obj in c[1000:] if obj.isdaughter==i0])
y = np.asarray([obj.vd for obj in c[1000:] if obj.isdaughter==i0])
z = np.asarray([obj.wb for obj in c[1000:]])
v = np.asarray([obj.wb for obj in c[1000:] if obj.isdaughter == i0])
print wbwb(par1, 2), np.mean(z*z)
print vdvb(par1, 1), np.mean(x*y)
print vbvb(par1, 1), np.mean(x*x)
print vb(par1, 1), np.mean(x)
print vd(par1, 1), np.mean(y)
print wb(par1, 1), np.mean(v)
val = scipy.stats.linregress(x, y)
print slope_vbvd(par1, celltype=i0), val[0]
