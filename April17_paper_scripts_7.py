#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import growth_simulation_dilution_asymmetric as g
import time
from scipy import stats
import scipy
import os.path
import matplotlib.pyplot as plt
tkw = dict(size=4, width=1.5)
colors = ['cornflowerblue', 'salmon']

td = 1.0
delta = 10.0
params = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 18), ('dt', 0.01),
            ('td', td), ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('CD', 1.0),
                 ('g2_std', 0.2), ('d_std', 0.1), ('g1_thresh_std', 0.1)])
celltype = ['Mothers', 'Daughters', 'Population']
num_rep = 4
N1 = 50001
# N=10
save_freq = 1000
X = [num_rep, N1, 2, 8]


def generate_paper_plots(path1, path2, cell_nums, par_vals, N, alpha=0.2):
    data = np.load(path1)  # symmetric division
    data_2 = np.load(path2)  # asymmetric division
    print data.shape, data_2.shape

    figs = []

    for i0 in range(cell_nums):
        tvec = np.linspace(1, N, N) * par_vals['nstep'] * par_vals['dt']
        indices = np.linspace(0.0, 100.0, 201)
        ln_inds = 2.0 * 10 ** (
            indices * 3.0 / np.amax(indices))  # gives a uniformly distributed set of points in a logx plot
        ln_inds = ln_inds.astype(int)

        figs.append([])
        fig = plt.figure(figsize=[10, 6])
        data_num = 1
        ax = plt.subplot(1, 1, 1)
        temp1 = data[:, :, i0, data_num]
        temp2 = data_2[:, :, i0, data_num]
        for i1 in range(5):
            tmp = data[i1, :, i0, data_num]
            plt.loglog(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        # plt.fill_between(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds] - np.std(temp1, axis=0)[ln_inds],
        #                  np.mean(temp1, axis=0)[ln_inds] + np.std(temp1, axis=0)[ln_inds], alpha=0.2)
        plt.loglog(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds], colors[0], label=celltype[i0] + r' mean, $r=1$', lw=5.0)
        for i1 in range(data_2.shape[0]):
            tmp = data_2[i1, :, i0, data_num]
            plt.loglog(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        # plt.fill_between(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds] - np.std(temp2, axis=0)[ln_inds],
        #                  np.mean(temp2, axis=0)[ln_inds] + np.std(temp2, axis=0)[ln_inds], alpha=0.2)
        plt.loglog(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds], colors[1], label=celltype[i0] + r' mean, $r=0.5$', lw=5.0)
        plt.xlabel(r'Time [$t_{db}$]', size=20)
        plt.ylabel(r'$\sigma (V_b)$', size=20)
        plt.xlim(xmin=0.0, xmax=par_vals['nstep']*par_vals['dt']*np.amax(ln_inds))
        plt.title(r'$\sigma (V_b)$ vs. time', size=20)
        plt.legend(loc=1)
        ax = g.nice_plot(ax, tkw)
        fig.savefig('./April17_paper_scripts_7_model_{0}_celltype_{1}_num_{2}_v2_sig_loglog_logpoints.eps'.format(str(par_vals['modeltype']),
                                                            str(i0), str(data_num)), dpi=fig.dpi, bbox_inches='tight')
        figs[i0].append(fig)
        del fig

        data_num = 0
        fig = plt.figure(figsize=[10, 6])
        ax = plt.subplot(1, 1, 1)
        temp1 = data[:, :, i0, data_num]
        temp2 = data_2[:, :, i0, data_num]
        for i1 in range(5):
            tmp = data[i1, :, i0, data_num]
            plt.loglog(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        plt.loglog(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds], colors[0], label=celltype[i0] + r' mean, $r=1$', lw=5.0)
        for i1 in range(data_2.shape[0]):
            tmp = data_2[i1, :, i0, data_num]
            plt.loglog(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        plt.loglog(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds], colors[1], label=celltype[i0] + r' mean, $r=0.5$', lw=5.0)
        plt.xlabel(r'Time [$t_{db}$]', size=20)
        plt.ylabel(r'$\langle V_b\rangle$', size=20)
        plt.xlim(xmin=0.0, xmax=par_vals['nstep']*par_vals['dt']*np.amax(ln_inds))
        plt.title(r'$\langle V_b\rangle$ vs. time', size=20)
        plt.legend(loc=1)
        ax = g.nice_plot(ax, tkw)
        fig.savefig('./April17_paper_scripts_7_model_{0}_celltype_{1}_num_{2}_v2_mean_loglog_logpoints.eps'.format(str(par_vals['modeltype']),
                                                            str(i0), str(data_num)), dpi=fig.dpi, bbox_inches='tight')
        figs[i0].append(fig)
        del fig

        data_num=1
        fig = plt.figure(figsize=[10, 6])
        ax=plt.subplot(1,1,1)
        temp1 = data[:, :, i0, data_num]/data[:, :, i0, 0]
        temp2 = data_2[:, :, i0, data_num]/data_2[:, :, i0, 0]
        for i1 in range(5):
            tmp = data[i1, :, i0, data_num]/data[i1, :, i0, 0]
            plt.semilogx(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        plt.semilogx(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds], colors[0], label=celltype[i0] + r' mean, $r=1$', lw=5.0)
        for i1 in range(data_2.shape[0]):
            tmp = data_2[i1, :, i0, data_num]/data_2[i1, :, i0, 0]
            plt.semilogx(tvec[ln_inds], tmp[ln_inds], alpha=alpha)
            del tmp
        plt.semilogx(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds], colors[1], label=celltype[i0] + r' mean, $r=0.5$', lw=5.0)
        plt.xlabel(r'Time [$t_{db}$]', size=20)
        plt.ylabel(r'$CV(V_b)$', size=20)
        plt.xlim(xmin=0.0, xmax=par_vals['nstep']*par_vals['dt']*np.amax(ln_inds))
        plt.title(r'$CV(V_b)$ vs. time', size=20)
        plt.legend(loc=1)
        ax = g.nice_plot(ax, tkw)
        fig.savefig('./April17_paper_scripts_7_model_{0}_celltype_{1}_num_{2}_v2_CV_logx_logpoints.eps'.format(str(par_vals['modeltype']),
                                                            str(i0), str(data_num)), dpi=fig.dpi, bbox_inches='tight')
        figs[i0].append(fig)
        del fig

        data_num = 5

        fig = plt.figure(figsize=[10, 6])
        ax = plt.subplot(1, 1, 1)
        temp1 = data[:, :, i0, data_num]
        temp2 = data_2[:, :, i0, data_num]
        mask1 = ~np.isnan(np.mean(temp1, axis=0))
        mask2 = ~np.isnan(np.mean(temp2, axis=0))
        ind1 = np.asarray([i1 for i1 in ln_inds if mask1[i1]])
        ind2 = np.asarray([i1 for i1 in ln_inds if mask2[i1]])
        plt.loglog(tvec[ind1], np.mean(temp1, axis=0)[ind1], colors[0], label=celltype[i0] + r' mean, $r=1$', lw=5.0)
        plt.fill_between(tvec[ind1], np.mean(temp1, axis=0)[ind1] - np.std(temp1, axis=0)[ind1],
                         np.mean(temp1, axis=0)[ind1] + np.std(temp1, axis=0)[ind1], alpha=0.2)
        plt.loglog(tvec[ind2], np.mean(temp2, axis=0)[ind2], colors[1], label=celltype[i0] + r' mean, $r=0.5$', lw=5.0)
        plt.fill_between(tvec[ind2], np.mean(temp2, axis=0)[ind2] - np.std(temp2, axis=0)[ind2],
                         np.mean(temp2, axis=0)[ind2] + np.std(temp2, axis=0)[ind2], alpha=0.2)
        print len(ind1), len(ind2)
        plt.xlabel(r'Time [$t_{db}$]', size=20)
        plt.ylabel(r'$\sigma (\log(V_b))$', size=20)
        plt.xlim(xmin=0.0, xmax=par_vals['nstep']*par_vals['dt']*np.amax(ln_inds))
        plt.title(r'$\sigma (\log(V_b))$ vs. time', size=20)
        plt.legend(loc=1)
        ax = g.nice_plot(ax, tkw)
        fig.savefig('./April17_paper_scripts_7_model_{0}_celltype_{1}_num_{2}_v2_sig_log_filled_logpoints.eps'.format(str(par_vals['modeltype']),
                                                            str(i0), str(data_num)), dpi=fig.dpi, bbox_inches='tight')
        figs[i0].append(fig)
        del fig

        ln_inds = np.linspace(0, 300, 301)
        ln_inds = ln_inds.astype(int)
        data_num = 5

        fig = plt.figure(figsize=[10, 6])
        ax=plt.subplot(1, 1, 1)
        temp1 = data[:, :, i0, data_num]
        temp2 = data_2[:, :, i0, data_num]
        plt.plot(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds], colors[0], label=celltype[i0] + r' $r=1$ mean', lw=5.0)
        plt.fill_between(tvec[ln_inds], np.mean(temp1, axis=0)[ln_inds] - np.std(temp1, axis=0)[ln_inds],
                         np.mean(temp1, axis=0)[ln_inds] + np.std(temp1, axis=0)[ln_inds], alpha=0.2)
        plt.plot(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds], colors[1], label=celltype[i0] + r' $r=0.5$ mean', lw=5.0)
        plt.fill_between(tvec[ln_inds], np.mean(temp2, axis=0)[ln_inds] - np.std(temp2, axis=0)[ln_inds],
                         np.mean(temp2, axis=0)[ln_inds] + np.std(temp2, axis=0)[ln_inds], alpha=0.2)
        plt.xlabel(r'Time [$t_{db}$]', size=20)
        plt.ylabel(r'$\sigma (\log(V_b))$', size=20)
        plt.xlim(xmin=0.0, xmax=par_vals['nstep']*par_vals['dt']*np.amax(ln_inds))
        plt.title(r'$\sigma (\log(V_b))$ vs. time', size=20)
        plt.legend(loc=1)
        ax = g.nice_plot(ax, tkw)
        fig.savefig('./April17_paper_scripts_7_model_{0}_celltype_{1}_num_{2}_v2_sig_log_filled.eps'.format(str(par_vals['modeltype']),
                                                            str(i0), str(data_num)), dpi=fig.dpi, bbox_inches='tight')
        tmp = np.mean(temp1, axis=0)
        mask = ~np.isnan(tmp)
        tmp1 = tvec[mask]
        tmp2 = tmp[mask]
        print tvec.shape, tmp.shape
        vals = scipy.stats.linregress(np.log(tmp1[:200]), np.log(tmp2[:200]))
        print celltype[i0], 'slope for first {0} gens:'.format(tvec[200]), vals[0]
        figs[i0].append(fig)
        del fig
    return figs

# params['modeltype'] = 18
# paths = ['March17_dilution_symmetric_3.npy', 'March17_dilution_symmetric_3_model_18_v1_asymm.npy']
# num = 2
# figs = generate_paper_plots(paths[0], paths[1], num, params)

# TO GENERATE SINGLE DATAPOINT
# num_rep = 20
# N1 = 20001
#
# params = dict([('num_s1', 500), ('nstep', 500), ('num_gen', 9), ('modeltype', 24), ('dt', 0.01), ('td', td),
#                  ('g1_std', 0.0), ('l_std', 0.0), ('g1_delay', 0.0), ('delta', delta), ('r', 0.5),
#                  ('r_std', 0.2), ('d_std', 0.0), ('g1_thresh_std', 0.0)])
#
# tmp1 = g.discr_gen(params)
# # This will initialize the subsequent simulations for this model
# tmp2 = g.starting_popn_seeded([obj for obj in tmp1 if obj.exists], params)
# # initial pop seeded from c
# tmp3, obs3 = g.discr_time_1(params, tmp2)
# a = np.zeros([3, 8])
# a1 = np.zeros([num_rep, N1, 3, 8])
# print len(tmp3)
#
# for i2 in range(3):  # full population included
#     if i2 <= 1:  # in this case we deal with mothers and daughters
#         temp2 = np.asarray([obj.vb for obj in tmp3 if obj.isdaughter == i2])
#         temp3 = np.asarray([obj.wb for obj in tmp3 if obj.isdaughter == i2])
#         a[i2, 0] = np.mean(temp2)
#         a[i2, 1] = np.std(temp2)
#         a[i2, 2] = np.mean(temp3)
#         a[i2, 3] = np.std(temp3)
#         a[i2, 4] = np.mean(np.log(temp2))
#         a[i2, 5] = np.std(np.log(temp2))
#         a[i2, 6] = np.mean(np.log(temp3))
#         a[i2, 7] = np.std(np.log(temp3))
#         del temp2, temp3
#     else:
#         temp2 = np.asarray([obj.vb for obj in tmp3])
#         temp3 = np.asarray([obj.wb for obj in tmp3])
#         a[i2, 0] = np.mean(temp2)
#         a[i2, 1] = np.std(temp2)
#         a[i2, 2] = np.mean(temp3)
#         a[i2, 3] = np.std(temp3)
#         a[i2, 4] = np.mean(np.log(temp2))
#         a[i2, 5] = np.std(np.log(temp2))
#         a[i2, 6] = np.mean(np.log(temp3))
#         a[i2, 7] = np.std(np.log(temp3))
#
# del tmp1, tmp2, tmp3, temp2, temp3
# for i0 in range(a1.shape[0]):
#     for i1 in range(a1.shape[1]):
#         a1[i0, i1, :, :] = a[:, :]
# srpath = './March17_dilution_symmetric_3_model_{0}_asymm_singlerep'.format(params['modeltype'])
# np.save(srpath, a1)
#
num_rep = 20
N1 = 20001
params['modeltype'] = 24  # Note this is inhibitor dilution.
paths = ['March17_dilution_symmetric_3_model_{0}_symm.npy'.format(params['modeltype']), 'March17_dilution_symmetric_3_model_{0}_asymm.npy'.format(params['modeltype'])]
# paths = ['March17_dilution_symmetric_3_model_{0}_symm.npy'.format(params['modeltype']), srpath+'.npy']
num = 3
figs = generate_paper_plots(paths[0], paths[1], num, params, N1)

# path = 'March17_dilution_symmetric_3_model_{0}_symm.npy'.format(par_vals['modeltype'])

params['modeltype'] = 4  # Note that this one is for initiator accumulation
paths = ['init_acc_evo_data_acq_model_{0}_symm.npy'.format(params['modeltype']), 'init_acc_evo_data_acq_model_{0}_asymm.npy'.format(params['modeltype'])]
num = 3
figs = generate_paper_plots(paths[0], paths[1], num, params, N1)
