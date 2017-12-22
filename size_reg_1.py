#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_bacteria_initiator as g
import time
from scipy import stats
import scipy


par1 = dict([('td', 1.0), ('modeltype', 2), ('g1_thresh_std', 0.02), ('num_gen1', 9), ('num_gen', 9), ('delta', 2.0), ('CD', 0.7),
             ('g2_std', 0.2), ('dt', 0.01), ('nstep', 500), ('num_s1', 500)])
print par1
c = []
temp1 = g.discr_gen(par1)
c.append(temp1)
del temp1
# This will initialize the subsequent simulations for this model
temp = g.starting_popn_seeded([obj for obj in c[0] if obj.exists], par1)
# initial pop seeded from c
temp1, obs1 = g.discr_time_1(par1, temp)
c.append(temp1)
del temp1, temp

for i0 in range(len(c)):
    temp = np.asarray([obj.vb for obj in c[i0] if obj.exists])
    temp1 = np.asarray([obj.wb for obj in c[i0] if obj.exists])
    temp2 = np.asarray([obj.vd for obj in c[i0] if obj.exists])
    temp3 = scipy.stats.linregress(temp, temp2)
    print np.mean(temp), g.vb_f(par1), np.mean(temp1), g.wb_f(par1), temp3[0]

fig=plt.figure(figsize=[6, 6])
plt.plot(obs1[1], np.log(obs1[0]), label='Log(N)')
x=obs1[1][100:]
vals = scipy.stats.linregress(x, np.log(obs1[0][100:]))
plt.plot(obs1[1][100:], vals[0]*x+vals[1], label='Slope = {0}'.format(np.round(vals[0], 3)))
plt.legend()
plt.xlabel('t')
plt.ylabel('Log(N)')
fig.savefig('./size_reg_1_logN.eps', bbox_inches='tight', dpi=fig.dpi)
print (vals[0]-np.log(2)/par1['td'])/vals[0]

del fig
fig=plt.figure(figsize=[6, 6])
plt.hexbin(temp, temp2, cmap="Purples", gridsize=60)
fig.savefig('./size_reg_1_vbvd.eps', bbox_inches='tight', dpi=fig.dpi)
del fig