#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import scipy
from scipy import stats
import seaborn as sns


f = open('./lab_meeting_figures/low_conc_investigation.txt', 'w')
filename = "Low concentration notes"
print >> f, 'Filename:', filename

par1 = dict([('g1_std', 0.0), ('g2_std', 0.14), ('g1_thresh_std', 0.0), ('nstep', 1200), ('dt', 0.01)
            , ('CD', 0.66), ('num_gen', 9), ('K', 10.0/0.66), ('td', 1.0), ('modeltype', 18),
             ('g1_delay', 0.0), ('l_std', 0.2), ('k_std', 0.0), ('d_std', 0.0), ('delta', 10.0),
             ('dead_mothers', False)])

models = [17, 18, 17, 18, 5, 10]
figs = []
l = 4
celltype = ['m', 'd']
meas = []
titles = []
for i0 in range(len(models)):
    fig = plt.figure(figsize=[13, 6])
    if i0 in [2, 3]:
        par1['d_std'] = 0.2
    elif i0 in [0, 1, 4, 5]:
        par1['d_std'] = 0.0
    print >> f, "model ", models[i0]
    print >> f, "parameters", par1
    print >> f, "Daughter theory slope " + str(
        np.round(g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['td'] * par1['g2_std']), 2))
    print >> f, "Mother theory slope " + str(
        np.round(g.slope_vbvd_m(par1, par1['g1_thresh_std'], par1['td'] * par1['g2_std']), 2))
    par1['modeltype'] = models[i0]
    c = g.discr_gen(par1)

    for k in range(2):
        ax = plt.subplot(1, 2, k + 1)
        ax.set_axis_bgcolor("w")
        num_gen = range(5)
        slopes = []
        for i in num_gen:
            x = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.mother.gen <= i]
            y = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.mother.gen <= i]
            slopes.append(scipy.stats.linregress(x, y))
            del x, y
        x = [obj.vb for obj in c[1000:] if obj.isdaughter == k]
        y = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        slopes.append(scipy.stats.linregress(x, y))
        num_gen.append(-1)

        x1 = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0]
        y1 = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0]
        print >> f, " Celltype", celltype[k], "percentage low conc =", len(x1) * 100.0 / len(x)
        plt.hexbin(x, y, gridsize=40)
        plt.scatter(x1, y1, marker='.', alpha=0.03, color='red')
        xmin, xmax, ymin, ymax = np.mean(x) - l * np.std(x), np.mean(x) + l * np.std(x), np.mean(y) - l * np.std(
            y), np.mean(y) + l * np.std(y)
        xvals = np.linspace(xmin, xmax, 10)
        i = 0
        for vals in slopes:
            plt.plot(xvals, vals[0] * xvals + vals[1],
                     label="Excl. mother gen>=" + str(num_gen[i]) + ", slope=" + str(np.round(vals[0], 2)))
            i += 1
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylim(ymin=ymin, ymax=ymax)
        titles.append("model " + str(models[i0]) + ' ' + celltype[
            k] + ", $\sigma_\Delta/\Delta=$"+str(par1['d_std'])+", $\sigma_i/\Delta=$"+str(par1['g1_thresh_std'])+", $\sigma_b/t_d=$"+str(par1['g2_std'])+", $\sigma_\lambda/\lambda=$"+str(par1['l_std']))
        plt.title(titles[2*i0+k])
        plt.legend(loc=2)

        plt.xlabel('$V_b$')
        plt.ylabel('$V_d$')
    figs.append(fig)
    meas.append(c)
    del c
    del fig

for i0 in range(len(figs)):
    filename = './lab_meeting_figures/model'+str(models[i0])+'num_'+str(i0)+'variable_mothergen.png'
    print >> f, "saved figure", filename
    figs[i0].savefig(filename, bbox_inches='tight', dpi=figs[i0].dpi)
del figs
figs = []
bound = [2.0, 2.5, 3.0]
bound = bound[::-1]
colors = ['red', 'blue', 'green', 'pink', 'cyan']
l = 5
print >>f, "Std deviation boundaries", bound
for i0 in range(len(models)):
    fig = plt.figure(figsize=[13, 6])
    par1['modeltype'] = models[i0]
    for k in range(2):
        ax = plt.subplot(1, 2, k + 1)
        ax.set_axis_bgcolor("w")
        slopes = []
        x = [obj.vb for obj in meas[i0][1000:] if obj.isdaughter == k]
        y = [obj.vd for obj in meas[i0][1000:] if obj.isdaughter == k]
        plt.hexbin(x, y, gridsize=40)
        xmin, xmax, ymin, ymax = np.mean(x) - l * np.std(x), np.mean(x) + l * np.std(x), np.mean(y) - l * np.std(
            y), np.mean(y) + l * np.std(y)
        xvals = np.linspace(xmin, xmax, 10)
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylim(ymin=ymin, ymax=ymax)
        xmean, ymean, xstd, ystd = np.mean(x), np.mean(y), np.std(x), np.std(y)
        for i1 in range(len(bound)):
            # print bound[i1], len(x)
            xmax, xmin, ymax, ymin = xmean + bound[i1]*xstd, xmean - bound[i1]*xstd, ymean + bound[i1]*ystd, ymean - bound[i1]*ystd
            temp = [obj for obj in meas[i0][1000:] if xmin <= obj.vb <= xmax and ymin <= obj.vd <= ymax and obj.isdaughter == k]
            x1 = [obj.vb for obj in temp]
            y1 = [obj.vd for obj in temp]
            plt.scatter(x1, y1, marker='.', alpha=0.05, color=colors[i1])
            del temp
            vals = scipy.stats.linregress(x1, y1)
            slopes.append(vals[0])
            plt.plot(xvals, vals[0] * xvals + vals[1],
                     label="Excl max$(|x-<x>|/\sigma_x$, $|x-<x>|/\sigma_x)>$" + str(bound[i1]) + ", slope=" + str(np.round(vals[0], 2)))
            del vals
        vals = scipy.stats.linregress(x, y)
        slopes.append(vals[0])
        plt.plot(xvals, vals[0] * xvals + vals[1],
                 label="Full population, slope=" + str(np.round(vals[0], 2)))

        plt.title(titles[2 * i0 + k])
        plt.legend(loc=2)
        plt.xlabel('$V_b$')
        plt.ylabel('$V_d$')
    figs.append(fig)
    print >>f, "Slopes", slopes[:-1]
    print >>f, "full population slope", slopes[-1]
    del fig

for i0 in range(len(figs)):
    filename = './lab_meeting_figures/model'+str(models[i0])+'num_'+str(i0)+'variable_population_sel.png'
    print >> f, "saved figure", filename
    figs[i0].savefig(filename, bbox_inches='tight', dpi=figs[i0].dpi)

# killing old mothers

del meas
del figs
figs = []
num_gen = range(2, 5)
num_gen.insert(0, 12)
par1['dead_mothers'] = True
models = [17, 18, 17, 18, 5, 10]
l = 2
for i0 in range(len(models)):
    fig = plt.figure(figsize=[13, 6])
    if i0 in [2, 3]:
        par1['d_std'] = 0.2
    elif i0 in [0, 1, 4, 5]:
        par1['d_std'] = 0.0
    print >> f, "model ", models[i0]
    print >> f, "parameters", par1
    print >> f, "Daughter theory slope " + str(
        np.round(g.slope_vbvd_func(par1, par1['g1_thresh_std'], par1['td'] * par1['g2_std']), 2))
    print >> f, "Mother theory slope " + str(
        np.round(g.slope_vbvd_m(par1, par1['g1_thresh_std'], par1['td'] * par1['g2_std']), 2))
    par1['modeltype'] = models[i0]
    slopes = [[], []]
    axes = []
    for i1 in range(len(num_gen)):
        par1['mother_gen'] = num_gen[i1]
        c = g.discr_gen(par1)
        for k in range(2):
            num_gen = range(5)
            if i1 == 0:
                ax = plt.subplot(1, 2, k + 1)
                ax.set_axis_bgcolor("w")
                axes.append(ax)
            plt.sca(axes[k])

            x = [obj.vb for obj in c[1000:] if obj.isdaughter == k]
            y = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
            slopes[k].append(scipy.stats.linregress(x, y))

            x1 = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0]
            y1 = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0]
            print >> f, " Celltype", celltype[k], ", generation of death:", num_gen[i1], ", percentage low conc =", len(x1) * 100.0 / len(x)

            if i1 == 0:
                plt.hexbin(x, y, gridsize=40)
                plt.scatter(x1, y1, marker='.', alpha=0.03, color='red')
            xmin, xmax, ymin, ymax = np.mean(x) - l * np.std(x), np.mean(x) + l * np.std(x), np.mean(y) - l * np.std(
                y), np.mean(y) + l * np.std(y)
            xvals = np.linspace(xmin, xmax, 10)
            i = 0
            for vals in slopes[k]:
                plt.plot(xvals, vals[0] * xvals + vals[1],
                         label="Gen of death =" + str(num_gen[i]) + ", slope=" + str(np.round(vals[0], 2)))
                i += 1
            plt.xlim(xmin=xmin, xmax=xmax)
            plt.ylim(ymin=ymin, ymax=ymax)
            titles.append("model " + str(models[i0]) + ' ' + celltype[
                k] + ", $\sigma_\Delta/Delta=$"+str(par1['delta'])+", $\sigma_i/\Delta=$"+str(par1['g1_thresh_std'])+", $\sigma_b/t_d=$"+str(par1['g2_std'])+", $\sigma_\lambda/\lambda=$"+str(par1['l_std']))
            plt.title(titles[2*i0+k])
            plt.legend(loc=2)

            plt.xlabel('$V_b$')
            plt.ylabel('$V_d$')
        del c
    figs.append(fig)
    del fig
for i0 in range(len(figs)):
    filename = './lab_meeting_figures/model'+str(models[i0])+'num_'+str(i0)+'dead_mothers.png'
    print >> f, "saved figure", filename
    figs[i0].savefig(filename, bbox_inches='tight', dpi=figs[i0].dpi)
f.close()
