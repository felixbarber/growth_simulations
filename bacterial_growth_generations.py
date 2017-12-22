# Needs to have the following packages already imported:
import numpy as np


def do_the_time_warp(x, y, z):
    return np.array([x+y+z**2])

par = dict([('td', 1.0), ('num_s', 100), ('Vo', 1.0), ('Wo', 1.0), ('std_iv', 0.1),
            ('std_iw', 0.1), ('std_it', 0.1)])
par['CD'] = 0.75*par['td']
par['k'] = 2**(1-par['CD']/par['td'])/par['CD']


def bact_growth(par1):
    vb = np.random.normal(loc=par['Vo'], scale=par['std_iv'], size=par['num_s'])
    wb = np.random.normal(loc=par['Wo'], scale=par['std_iv'], size=par['num_s'])
    delta = 2 ** (-par['CD'] / par['td'])
    b = 1.0
    num_gen = 14

    for i in range(num_gen):
        if par1['g1_thresh_std'] != 0:  # gaussian noise in the abundance of initiator required to cause initiation
            noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], len(vb))
        else:
            noise_thr = 0.0
        if par1['g1_std'] != 0:
            noise_g1 = np.random.normal(0.0, par1['g1_std'], len(vb))
            # generate and retain the time additive noise in the first part of the cell cycle.
        else:
            noise_g1 = 0.0
        if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
            noise_g2 = np.random.normal(0.0, par1['g2_std'], len(vb))
        else:
            noise_g2 = 0.0
        vi = ((delta + noise_thr - wb) / b + vb) * np.exp(
            noise_g1 * np.log(2) / par['td'])
        vd = vi * np.exp((par['CD'] + noise_g2) * np.log(2) / par['td'])
        wd = b * (vd - vi)
        if i < num_gen-1:
            vb = np.empty(len(vd) * 2)
            wb = np.empty(len(vd) * 2)
            for j in range(2):
                vb[j*len(vd):(j+1)*len(vd)] = 0.5 * vd[:]
                wb[j*len(vd):(j+1)*len(vd)] = 0.5 * wd[:]
            del vi, vd, wd

    return vb, vd