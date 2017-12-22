# Needs to have the following packages already imported:
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt

def do_the_time_warp(x,y,z):
    return np.array([x+y+z**3])

par = dict([('td', 1.0), ('num_s', 100), ('Vo', 1.0), ('std_iv', 0.1),
            ('std_iw', 0.1), ('delta_v', 1.0), ('k', 1.0)])


class Mother:  # Define the class of mother cells

    td = par['td']  # mass doubling time for cells
    K = par['k']  # production rate of Whi5 (produces one unit of Whi5 in time taken for cell mass to double).
    cellCount = 0  # total number of daughter cells

    def __init__(self, vb, whi5, tb, mother=None, growth_pol=0):
        # produces a new instance of the mother class. Only instance variables
        # initially are the whi5 abundance at birth, the volume at birth, the time of birth and it points to the mother.
        self.vb = vb
        self.wb = whi5
        self.tb = tb
        self.mother = mother  # weakly references the mother cell
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.growth_pol = growth_pol  # indexes which kind of growth policy this cell will implement
        self.exists = True  # indexes whether the cell exists currently
        self.R = None
        self.t_delay = None
        self.noise_thr = None
        self.noise_g1 = None
        self.vs = None
        self.noise_g2 = None
        self.td = None
        self.vd = None
        self.wd = None
        self.t_grow = None
        self.t_div = None

        Mother.cellCount += 1

    def grow(self, inp_par):
        self.R = inp_par['r']
        self.t_delay = inp_par['t_delay']
        if self.growth_pol == 0:  # Whi5 dilution model

            # Calculate the volume of this cell at start, volume at division, and whi5 at division.
            if inp_par['thr_std'] != 0:
                self.noise_thr = np.random.normal(0.0, inp_par['thr_std'], 1)[0]
            else:
                self.noise_thr = 0.0
            if inp_par['g1_std'] != 0:
                self.noise_g1 = np.random.normal(0.0, inp_par['g1_std'], 1)[0]
                # generate and retain the noise in G1 for each instance of the mother class
            else:
                self.noise_g1 = 0.0
            # note that noise is measured as a fraction of growth rate.
            self.vs = (self.wb / (1 + self.noise_thr)) * np.exp(
                self.noise_g1 * np.log(2) / self.td) * 2 ** self.t_delay
            if inp_par['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                self.noise_g2 = np.random.normal(0.0, inp_par['g2_std'], 1)[0]
            else:
                self.noise_g2 = 0.0
            self.vd = self.vs * (1.0 + self.R) * np.exp(self.noise_g2 * np.log(2) / self.td)
            # self.wd = self.wb + np.maximum(np.log(self.vd / self.vs) * self.K / np.log(2), 0.0)
            self.wd = self.wb + self.K * (np.log(1.0 + self.R) * self.td / np.log(2.0) + self.noise_g2)
            self.t_grow = np.maximum(np.log(self.vd / self.vb) * self.td / np.log(2), 0.0)
            self.t_div = self.tb + self.t_grow


class Daughter:  # Define the class of mother cells

    td = par['td']  # mass doubling time for cells
    K = par['k']   # production rate of Whi5 (produces one unit of Whi5 in time taken for cell mass to double).
    cellCount = 0  # total number of daughter cells

    def __init__(self, vb, whi5, tb, mother=None, growth_pol=0):
        # produces a new instance of the mother class. Only instance variables
        # initially are the whi5 abundance at birth, the volume at birth, the time of birth and it points to the mother.
        self.vb = vb
        self.wb = whi5
        self.tb = tb
        self.mother = mother  # weakly references the mother cell
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.growth_pol = growth_pol  # indexes which kind of growth policy this cell will implement
        self.exists = True  # indexes whether the cell exists currently
        self.R = None
        self.t_delay = None
        self.noise_thr = None
        self.noise_g1 = None
        self.vs = None
        self.noise_g2 = None
        self.vd = None
        self.wd = None
        self.t_grow = None
        self.t_div = None

        Daughter.cellCount += 1

    def grow(self, inp_par):
        self.R = inp_par['r']
        self.t_delay = inp_par['t_delay']
        if self.growth_pol == 0:  # Whi5 dilution model

            # Calculate the volume of this cell at start, volume at division, and whi5 at division.
            if inp_par['thr_std'] != 0:
                self.noise_thr = np.random.normal(0.0, inp_par['thr_std'], 1)[0]
            else:
                self.noise_thr = 0.0
            if inp_par['g1_std'] != 0:
                self.noise_g1 = np.random.normal(0.0, inp_par['g1_std'], 1)[0]
                # generate and retain the noise in G1 for each instance of the mother class
            else:
                self.noise_g1 = 0.0
            # note that noise is measured as a fraction of growth rate.
            self.vs = (self.wb / (1 + self.noise_thr)) * np.exp(
                self.noise_g1 * np.log(2) / self.td) * 2 ** self.t_delay
            if inp_par['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                self.noise_g2 = np.random.normal(0.0, inp_par['g2_std'], 1)[0]
            else:
                self.noise_g2 = 0.0
            self.vd = self.vs * (1.0 + self.R) * np.exp(self.noise_g2 * np.log(2) / self.td)
            # self.wd = self.wb + np.maximum(np.log(self.vd / self.vs) * self.K / np.log(2), 0.0)
            self.wd = self.wb + self.K * (np.log(1.0 + self.R) * self.td / np.log(2.0) + self.noise_g2)
            self.t_grow = np.maximum(np.log(self.vd / self.vb) * self.td / np.log(2), 0.0)
            self.t_div = self.tb + self.t_grow