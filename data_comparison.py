#!/usr/bin/env python

import growth_simulation_dilution_asymmetric as g
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy.matlib
import seaborn as sns

#  import observables calculated based on Ilya's data

data = np.load('growth_condition_data.npy')
pop_data = np.load('growth_condition_data_pop.npy')
NAMES = list(['Dip Glucose', 'Dip Galactose', 'Dip Glycerol', 'Dip Low glucose',
              'Dip Raffinose', 'Hap Glucose', 'Hap Galactose', 'Hap Glycerol',
              'Hap Low glucose'])
with open('growth_condition_data_labels.csv', 'rb') as f:
    reader = csv.reader(f)
    labels = list(reader)
del reader
labels = labels[0]

# First we just plot the variation with k for each dataset
num_cond = pop_data.shape[0]
celltype = ['Mothers', 'Daughters']


def corr_d(td, cd, s_cd, D, s_i):
    l = np.log(2) / td
    r = np.exp(l*cd)-1
    a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
    vbvb = ((1+r)**2*np.exp(2*(l*s_cd)**2)-2*(1+r)*np.exp(0.5*(l*s_cd)**2)+1)*(3*a*D**2/(2-a)+s_i**2)
    vb = D * ((1 + r) * np.exp(0.5 * (l * s_cd) ** 2) - 1)
    vd = (1 + r - np.exp(0.5 * (l * s_cd) ** 2)) * np.exp(0.5 * (l * s_cd) ** 2) * 2 * D
    vdvb = D ** 2 * (1 + r) * np.exp(0.5 * (l * s_cd) ** 2) * (np.exp(0.5 * (l * s_cd) ** 2) * ((1 + r) + 1 / (1 + r)) - 2) * (
        2 + 2 * a) / (2 - a)
    slope=(vdvb-vb*vd)/(vbvb-vb**2)
    return vb, vd, vbvb, vdvb, slope


def corr_m(td, cd, s_cd, D, s_i):
    l = np.log(2) / td
    r = np.exp(l*cd)-1
    a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
    vbvb = 3 * D ** 2 * a / (2 - a) + s_i ** 2
    vb = D
    vd = 2*D*np.exp((l*s_cd)**2)
    vdvb = np.exp((l*s_cd)**2)*(2+2*a)/(2-a)*D**2
    slope = (vdvb-vb*vd)/(vbvb-vb**2)
    return vb, vd, vbvb, vdvb, slope


def noiseless_adder_slopes(td, cd, s_cd, delta, s_i):
    d = corr_d(td, cd, s_cd, delta, s_i)
    m = corr_m(td, cd, s_cd, delta, s_i)
    return m[4], d[4]


def standard_error(params):  # returns the standard error for a specific set of parameters by simulating 10^6 different
        # value sets based on statistically inferred parameter variability around these measured values.
        # Uses estimates of sample error based on the sample values.
    num = 10**6
    td = np.random.normal(params['td'], params['td_err'], 10**6)
    cd = np.random.normal(params['CD'], params['CD_err'], 10**6)
    s_cd = np.random.normal(params['s_cd'], params['s_cd_err'], 10**6)
    delta = np.random.normal(params['delta'], params['delta_err'], 10**6)
    s_i = params['s_i']  # assumes that you have already measured the optimal error in the timing threshold
    [slopes_m, slopes_d] = noiseless_adder_slopes(td, cd, s_cd, delta, s_i)

    return [np.std(slopes_m), np.std(slopes_d)]


def optimal_slope(params, data, ind, data_fit, start_ind):
    g1_std = np.linspace(0.0, 0.5 * params['delta'],
                         200)  # a generous range of variability for the noise in the Whi5 threshold
    td = params['td']
    cd = params['CD']
    s_cd = params['s_cd']
    delta = params['delta']
    slope = np.empty([2, len(g1_std)])
    slope[0, :], slope[1, :] = noiseless_adder_slopes(td, cd, s_cd, delta, g1_std)
    diff_fit = slope[start_ind:data_fit.shape[0]+start_ind, :] - np.transpose(np.matlib.repmat(data_fit[:, ind, 0], slope.shape[1],1))
    norm = LA.norm(diff_fit, axis=0)
    index = np.argmin(norm)
    diff = np.empty(slope.shape)
    diff[0, :] = slope[0, :] - data[0, j, 0]
    diff[1, :] = slope[1, :] - data[1, j, 0]

    return index, g1_std[index], slope[:, index], diff, g1_std


params = dict([])
fig = plt.figure(figsize=[15, 17])
plt.title('Whi5 noiseless adder - measured slope for variable Whi5 threshold noise $\sigma_i$')
sigma_i = np.empty([3,num_cond])
slope_opt = np.empty([3, 2, num_cond])
slope_opt_err = np.empty([3, 2, num_cond])
for j in range(num_cond):
    params['CD'] = pop_data[j, 11]
    params['s_cd'] = pop_data[j, 13]
    params['td'] = pop_data[j, 7]
    params['delta'] = data[0, j, 4]
    params['delta_err'] = data[0, j, 14]
    params['td_err'] = pop_data[j, 16]
    params['CD_err'] = pop_data[j, 14]
    params['s_cd_err'] = pop_data[j, 15]
    index = np.empty([3])
    data_m = np.empty([1, data.shape[1], data.shape[2]])
    data_m[0, j, :] = data[0, j, :]
    data_d = np.empty([1, data.shape[1], data.shape[2]])
    data_d[0, j, :] = data[1, j, :]
    index[0], sigma_i[0, j], slope_opt[0, :, j], diff, g1_std = optimal_slope(params, data, j, data, 0)
    index[1], sigma_i[1, j], slope_opt[1, :, j], diff1, g1_std = optimal_slope(params, data, j, data_m, 0)
    index[2], sigma_i[2, j], slope_opt[2, :, j], diff2, g1_std = optimal_slope(params, data, j, data_d, 1)
    params['s_i'] = sigma_i[0, j]
    slope_opt_err[0, :, j] = standard_error(params)
    params['s_i'] = sigma_i[1, j]
    slope_opt_err[1, :, j] = standard_error(params)
    params['s_i'] = sigma_i[2, j]
    slope_opt_err[2, :, j] = standard_error(params)
    plt.subplot(3, 3, j + 1)
    for k in range(diff.shape[0]):
        plt.plot(g1_std / params['delta'], diff[k, :], label=celltype[k] + ' model - data')
    plt.xlabel('Whi5 threshold noise $\sigma_i /\Delta$')
    plt.ylabel('Model slope - expt slope (unitless)')
    plt.plot((g1_std[index[0]]/params['delta'], g1_std[index[0]]/params['delta']), (diff[0, index[0]], diff[1, index[0]]), 'c.', label='MD fitted value =' + str(round(sigma_i[0, j], 0)))
    plt.plot((g1_std[index[1]]/params['delta'], g1_std[index[1]]/params['delta']), (diff1[0, index[1]], diff1[1, index[1]]), 'g.', label='M fitted value =' + str(round(sigma_i[1, j], 0)))
    plt.plot((g1_std[index[2]]/params['delta'], g1_std[index[2]]/params['delta']), (diff2[0, index[2]], diff2[1, index[2]]), 'm.', label='D fitted value =' + str(round(sigma_i[2, j], 0)))
    plt.legend()
    plt.title('Slope difference ' + NAMES[j])
plt.suptitle('Slope fitting Noiseless Whi5 adder model')
fig.savefig('variable_si_model_fitting.eps', bbox_inches='tight', dpi=fig.dpi)

fig2 = plt.figure(figsize=[15, 7])
font = {'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
for k in range(2):
    plt.subplot(1, 2, k+1)
    plt.errorbar(np.linspace(1.0, 9.0, 9), data[k, :, 0], yerr=data[k, :, 1], fmt='ro', ms=10, label='Experimental $V_b$ $V_d$ slope')
    plt.errorbar(np.linspace(1.0, 9.0, 9), slope_opt[0, k, :], yerr=slope_opt_err[0, k, :], fmt='c,', ms=10, label='MD fitted $V_d$ $V_b$ slope')
    plt.errorbar(np.linspace(1.0, 9.0, 9), slope_opt[1, k, :], yerr=slope_opt_err[1, k, :], fmt='g,', ms=10, label='Mother fitted $V_d$ $V_b$ slope')
    plt.errorbar(np.linspace(1.0, 9.0, 9), slope_opt[2, k, :], yerr=slope_opt_err[2, k, :], fmt='m,', ms=10, label='Daughter fitted $V_d$ $V_b$ slope')
    #plt.plot(np.linspace(1.0, 9.0, 9), slope_opt[k, :], 'b.', ms=10,
    #            label='Fitted $V_d$ $V_b$ slope')
    plt.legend(prop={'size': 12})
    plt.title(celltype[k], size=16, weight="bold")
    plt.ylabel("CV=$\sigma/\mu$", size=16, weight="bold")
    plt.margins(0.2)
    plt.xticks(np.linspace(1.0, 9.0, 9), NAMES, rotation='vertical')
plt.suptitle('Slope fitting for Noiseless Whi5 adder model')
fig2.savefig('sigma_i_fittedslopes_md_noiselesswhi5adder.eps', bbox_inches='tight', dpi=fig2.dpi)

s_cd = np.linspace(0.0, 0.3,
                     31)  # a generous range of variability for the noise in the Whi5 threshold
g1_std = np.linspace(0.0, 0.3,
                     31)  # a generous range of variability for the noise in the Whi5 threshold
td = 1.0
cd = 0.75*td
delta = cd
slope = np.empty([2, len(g1_std), len(s_cd)])
for i in range(len(s_cd)):
    slope[0, :, i], slope[1, :, i] = noiseless_adder_slopes(td, cd, s_cd[i], delta, g1_std)
for i in range(2):
    fig = (plt.figure(figsize=[16, 15]))
    sns.heatmap(slope[i, ::-2, ::2], xticklabels=np.around(s_cd[::2], decimals=2),
                yticklabels=np.around(g1_std[::-2], decimals=2), annot=True)
    plt.xlabel('C+D timing noise $\sigma_{C+D}$', size=20)
    plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
    plt.title(celltype[i], size=20)
    fig.savefig('function_test_'+str(i)+'.eps', bbox_inches='tight', dpi=fig.dpi)