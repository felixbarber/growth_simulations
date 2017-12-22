#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy.matlib
import seaborn as sns

#  import observables calculated based on Ilya's data

# "$V_d$ $V_b$ slope 0","$V_d$ $V_b$ slope std error 1","$<r>$ 2","CV $r$ 3","$<V_b>$ 4","CV $V_b$ 5","$<V_d>$ 6",\
# "CV $V_d$ 7","Mass doubling time $<t_d>$ 8","CV Mass doubling time $t_d$ 9","$<t_{G1}>$ 10","CV $t_{G1}$ 11",\
# "$<t_{bud}>$ 12","CV $t_{bud}$ 13","SEM $V_b$ 14","Division time $<t_{div}>$ 15","CV Division time $t_{div}$ 16",\
# "Growth Rate $<\lambda>$ 17","CV Growth Rate $\lambda$ 18","std(t_{bud})/mean(t_doub) 19","std(Tdiv)/mean(t_doub) 20"

data = np.load('growth_condition_data.npy')
# pop_data = np.load('growth_condition_data_pop.npy')
NAMES = list(['Dip Glu', 'Dip Gal', 'Dip Gly', 'Dip LGl',
              'Dip Raf', 'Hap Glu', 'Hap Gal', 'Hap Gly',
              'Hap LGl'])
with open('growth_condition_data_labels.csv', 'rb') as f:
    reader = csv.reader(f)
    labels = list(reader)
del reader
labels = labels[0]
# with open('growth_condition_data_pop_labels.csv', 'rb') as f:
#     reader = csv.reader(f)
#     labels_pop = list(reader)
# del reader
# labels_pop = labels_pop[0]

# First we just plot the variation with k for each dataset
print data.shape
num_cond = data.shape[1]
celltype = ['M', 'D', 'P']
k = 0
inds = [0,15, 20, 12, 19, 17, 18, 10]
print NAMES[k]
for j in range(3):
    for ind in inds:
        print celltype[j], np.round(data[j, k, ind], 3), labels[ind]
    print celltype[j], np.round(data[j, 2, ind]/(1+data[j, 2, ind]))
    print celltype[j], " Minimum ", labels[12], np.round(np.amin(data[j, :4, 12]/data[j, :4, 8]), 2)
    print celltype[j], " Maximum ", labels[12], np.round(np.amax(data[j, :4, 12]/data[j, :4, 8]), 2)
    print celltype[j], " Minimum ", labels[0], np.round(np.amin(data[j, :4, 0]), 2)
    print celltype[j], " Maximum ", labels[0], np.round(np.amax(data[j, :4, 0]), 2)
    i1 = 19
    print celltype[j], " Minimum ", labels[i1], np.round(np.amin(data[j, :4, i1]), 2)
    print celltype[j], " Maximum ", labels[i1], np.round(np.amax(data[j, :4, i1]), 2)
    i1 = 18
    print celltype[j], " Minimum ", labels[i1], np.round(np.amin(data[j, :4, i1]), 2)
    print celltype[j], " Maximum ", labels[i1], np.round(np.amax(data[j, :4, i1]), 2)
    i1 = 21
    print celltype[j], " Minimum ", labels[i1], np.round(np.amin(data[j, :4, i1]), 2)
    print celltype[j], " Maximum ", labels[i1], np.round(np.amax(data[j, :4, i1]), 2)
# print " Minimum ", labels[12], np.round(np.amin(data[:, :4, 12] / data[:, :4, 8]), 2)
# print " Maximum ", labels[12], np.round(np.amax(data[:, :4, 12] / data[:, :4, 8]), 2)
# print " Minimum ", labels[0], np.round(np.amin(data[:, :4, 0]), 2)
# print " Maximum ", labels[0], np.round(np.amax(data[:, :4, 0]), 2)
# i1 = 19
# print " Minimum ", labels[i1], np.round(np.amin(data[:, :4, i1]), 2)
# print " Maximum ", labels[i1], np.round(np.amax(data[:, :4, i1]), 2)

# for j in range(2):
#     for k in range(num_cond):
#         l1 = 8
#         l2 = 12
#         print NAMES[k], np.round(data[j, k, l2]/data[j, k, l1], 2), celltype[j], labels[l2]+'/'+labels[l1]
# for k in range(num_cond):
#     l1 = 7
#     l2 = 11
#     print NAMES[k], np.round(pop_data[k, l2]/pop_data[k, l1], 2), 'pop', labels_pop[l2]+'/'+labels_pop[l1]
