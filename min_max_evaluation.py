#!/usr/bin/env python

import scipy
from scipy import optimize
import growth_simulation_dilution_asymmetric as g
import numpy as np
import time

vec_list = ['CD', 'K', 'g1_std', 'g2_std', 'l_std', 'k_std']
def model5_slope_d(vec):  # x is a variable of length N = 6
    par_dict = dict([('modeltype', 5)])  # define the dictionary of terms to be passed to the function from g
    par_dict['td'] = 1.0  # doubling time. Varying this doesn't change anything, since all other variables
    # are defined relative to the doubling time.
    par_dict['CD'] = vec[0]  # CD period as fraction of td -- sims
    par_dict['K'] = vec[1]  # Whi5 synthesis rate in volume per doubling time -- sims
    g1_std = vec[2]  # Noise in Whi5 Start threshold in units of volume -- sims
    g2_std = vec[3]  # unit less fraction of td, NOT of cd. -- sims
    par_dict['l_std'] = vec[4]  # unit less fraction of growth rate -- sims
    par_dict['k_std'] = vec[5]  # unit less fraction of synthesis rate K -- sims
    val = g.slope_vbvd_func(par_dict, g1_std, g2_std * par_dict['td'])
    # returns the predicted slope for these values
    print g.wbwb_p(par_dict, g1_std, g2_std * par_dict['td'])
    print g.vb_func(par_dict, g1_std, g2_std * par_dict['td'])
    print g.vd_func(par_dict, g1_std, g2_std * par_dict['td'])
    print g.vdvb_func(par_dict, g1_std, g2_std * par_dict['td'])
    print g.vbvb_func(par_dict, g1_std, g2_std * par_dict['td'])
    td = par_dict['td']
    cd = par_dict['CD'] * par_dict['td']  # units of time
    l = np.log(2) / td
    s_l = par_dict['l_std'] * l
    print g.exp_xy(l, s_l, cd, g2_std * par_dict['td'])
    print g.y_exp_xy(l, s_l, cd, g2_std * par_dict['td'])
    print g.yy_exp_xy(l, s_l, cd, g2_std * par_dict['td'])
    return val

bounds = [(0.5, 0.9), (0.5, 1.5), (0.01, 0.4), (0.01, 0.4), (0.01, 0.4), (0.01, 0.4)]
par_vals = list([])
num_discr = 10
for bound in bounds:
    par_vals.append(np.linspace(bound[0], bound[1], num_discr))

# fn_vals = np.zeros([num_discr, num_discr, num_discr, num_discr, num_discr, num_discr])
# x = np.zeros(len(bounds))
# for i1 in range(num_discr):
#     tic = time.clock()
#     x[0] = par_vals[0][i1]
#     for i2 in range(num_discr):
#         x[1] = par_vals[1][i2]
#         for i3 in range(num_discr):
#             x[2] = par_vals[2][i3]
#             for i4 in range(num_discr):
#                 x[3] = par_vals[3][i4]
#                 for i5 in range(num_discr):
#                     x[4] = par_vals[4][i5]
#                     for i6 in range(num_discr):
#                         x[5] = par_vals[5][i6]
#                         val = model5_slope_d(x)
#                         if not(np.isnan(val)):
#                             fn_vals[i1, i2, i3, i4, i5, i6] = val
#                         else:
#                             fn_vals[i1, i2, i3, i4, i5, i6] = 1.0
#     print "iteration number ", i1, " time taken ", time.clock()-tic
# np.save('model5_minmax_slopes', fn_vals)

v1 = np.array([0.75, 1.0, 0.2, 0.4, 0.4, 0.4])
print model5_slope_d(v1)
# std_init = 0.0
# x0 = np.array([0.75, 1.0, std_init, std_init, std_init, std_init])  # initialization vector for optimization function

# if len(bounds) == len(x0):
#     x, nfeval, rc = scipy.optimize.fmin_tnc(model5_slope_d, x0, approx_grad=True)
# else:
#     raise ValueError('Typo in length of bounds')
# np.save('model5_minimum', x)
# print "number of function evaluations", nfeval
# print "coordinates", x, "value", model5_slope_d(x)
#
# res = scipy.optimize.basinhopping(model5_slope_d, x0, niter=100, T=0.5, stepsize=0.1)
# np.save('model5_minimum_basin', res.x)
# print "num it", res.nit
# print "basin coordinates", res.x, "value", model5_slope_d(res.x)
