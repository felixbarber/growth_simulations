# Needs to have the following packages already imported:
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt
import copy


def do_the_time_warp(x,y,z):
    return np.array([x+y+z**4])

########################################################################################################################
# DISCRETIZED TIME SIMULATIONS -- slow bacterial growth
########################################################################################################################

par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('wd', 1.0), ('wm', 1.0), ('std_v', 0.2), ('std_w', 0.2)])
list1 = [1, 2]  # noisy initiation
list2 = [2]  # no shrinking


class Cell(object):
    # 'common base class for all mother cells of different growth policies'
    cellCount = 0  # total number of cells
    negcellCount = 0  # number of cells with a cell division time sufficiently low relative to their time of birth that
    # they are neglected.
    b = 1.0

    def __init__(self, vb, w, tb, mother=None):
        # produces a new instance of the mother class. Only instance variables
        # initially are the initiator abundance at birth (w), the volume at birth,
        # the time of birth and it points to the mother.
        self.vb = vb
        self.wb = w
        self.tb = tb
        self.mother = mother  # references the mother cell
        self.daughter = []  # weakly references the daughter cells
        self.exists = True  # indexes whether the cell exists currently. Use in combination with should_div to check
        # whether some small cells are managing to skip division.
        self.should_divide = False  # indexes whether a cell should have divided during the simulation (quality control)
        self.gen_m = 0
        self.gen_d = 0

        Cell.cellCount += 1

    def grow(self, par1):  # integration model for accumulation of initiator protein. In slow growth case.
        # necessary variables:
        # par1['delta']: amount of adder required to initiate Start. This will be necessary for all models.
        # Calculate the volume of this cell at initiation and division.

        l = np.log(2)/par1['td']

        # G1 Part of the cell cycle

        if par1['modeltype'] in list1 and par1['g1_thresh_std'] != 0:  # noisy initiation
            noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
        else:
            noise_thr = 0.0
        if par1['modeltype'] in list2:  # cell has a minimum volume at initiation
            self.vi = max(self.vb - self.wb + par1['delta']*(1+noise_thr), self.vb)
        else:
            self.vi = self.vb - self.wb + par1['delta']*(1+noise_thr)
        del noise_thr

        # G2 part of the cell cycle

        self.vd = self.vi - 1  # ensures that the following code is run at least once.

        # prevents shrinking in cells in list8. If we do this for all models then you're trying to make a positive * neg
        # = a positive and it takes forever.
        if par1['modeltype'] in list2:
            while self.vd <= self.vi:  # resample until we get a volume at division which is greater than that at birth.
                if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                    noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
                else:
                    noise_g2 = 0.0
                self.vd = self.vi * np.exp((par1['CD'] + noise_g2) * par1['td'] * l)
        else:
            if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
            else:
                noise_g2 = 0.0
            self.vd = self.vi * np.exp((par1['CD'] + noise_g2) * par1['td'] * l)
        self.wd = self.vd - self.vi  # all initiator destroyed on passage through Start
        self.t_grow = np.log(self.vd / self.vb) / l
        self.t_bud = np.log(self.vd/self.vi) / l
        self.t_div = self.tb + max(self.t_grow, 0.0)

        Cell.cellCount += 1


def starting_popn(par1):  # note that this is only for one cell type
    # To clarify we first set the initial condition for the simulation.
    l = np.log(2)/par1['td']
    if par1['modeltype'] in [1, 2]:  # seed cells with vb and wb around the analytically predicted means
        s_cd = par1['g2_std']*par1['td']
        cd = par1['CD']*par1['td']
        wd = par1['delta']*(np.exp(l*cd+0.5*(l*s_cd)**2)-1)
        vd = par1['delta'] * np.exp(l*cd+0.5*(l*s_cd)**2)

    # par=dict([('dt', 1),('nstep',100), ('td', 90), ('num_s', 100),('Vo',1.0),('std_iv',0.1),('std_iw',0.1)])
    # Initialise simulation with a normal distribution of cell sizes and Whi5 abundances.
    v_init_d = np.random.normal(loc=vd, scale=par['std_v']*vd, size=par['num_s'])
    w_init_d = np.random.normal(loc=wd, scale=par['std_w']*wd, size=par['num_s'])
    t_div = np.random.uniform(low=0.0, high=1.0, size=par['num_s'])
    # gives the fraction of the cell's doubling time which this cell has passed through.

    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []

    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        c.append(Cell(v_init_d[i], w_init_d[i], 0))
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div-c[-1].t_grow
    del v_init_d, w_init_d
    return c


def starting_popn_seeded(c, par1):
    # c is the set of "existent" cells at the end of the previous simulation, done so as to yield a converged
    # distribution of cells

    indices = np.random.randint(low=0, high=len(c), size=par1['num_s1'])
    temp = [c[ind] for ind in indices]
    val = []
    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(copy.copy(obj))

    t_div = np.random.uniform(low=0.0, high=1.0, size=par1['num_s1'])
    # gives the fraction of the cell's doubling time which this cell has passed through.

    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.

    for i0 in range(par1['num_s1']):
        val[i0].t_div = max(t_div[i0] * val[i0].t_grow, par1['dt']*par1['td']*1.01)  # to ensure that this cell gets
        # detected in the iteration through calculating next generation cells.
        # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        val[i0].tb = val[i0].t_div-val[i0].t_grow
    return val


def next_gen(index, f, t, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    frac = 0.5

    f.append(Cell(frac*(f[index].vd), frac*f[index].wd, t, mother=weakref.proxy(f[index])))  # generates
    # daughter cell 1.
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter.append(weakref.proxy(f[-1]))  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.

    f.append(Cell(frac * (f[index].vd), frac * f[index].wd, t, mother=weakref.proxy(f[index])))  # generates
    # daughter cell 2.
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter.append(weakref.proxy(f[-1]))  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.
    f[index].exists = False  # track that this cell no longer "exists".
    return f


def discr_time_1(par1, starting_pop):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1 and a starting population of cells.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * par1['td'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep + 1):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_pop[:]
    num_cells[0] = len(c)
    for i in range(len(c)):
        if c[i].t_div < np.amax(tvec):  # ensures we never consider cells which would go into the (nstep+2)th bin, since
            # there is none in this setup
            td_ind = np.searchsorted(tvec, np.array(c[i].t_div), side='left', sorter=None)
            # left means that e.g. (n-1)*dt<x<=n*dt goes to the nth bin.
            # nstep*dt<x means that the index is (nstep+1).
            div_times[td_ind].append(i)
            # a cell is picked to divide in the timestep after it divides, i.e. 11 if divides in the 11th time interval
            del td_ind
    # Now we begin iterating through the time values in tvec
    for i in range(nstep + 1):
        for index in div_times[i]:  # the presence of indices in the 0th timestep means that the seeding cells grew
            # negatively. Statistically unlikely but may happen for one or two cells.

            # Double check that these indices have been allocated correctly: c[index].t_div should be in the interval
            # (tvec[i-1], tvec[i]]
            if i != 0:
                if tvec[i - 1] >= c[index].t_div or tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i + 1]
                    raise ValueError('The cell division orders are wrong')
            elif i == 0:
                if tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i + 1]
                    raise ValueError('The cell division orders are wrong')

            c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
            for j in range(2):  # Set the time of division for the two new cells produced by next_gen
                t_div = c[len(c) - 1 - j].t_div
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the time frame of our
                    # simulation
                    c[len(c) - 1 - j].should_div = True  # this cell should for sure divide at some stage within the sim
                    # simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    if td_ind < i:
                        raise ValueError('Cells are falling behind')
                    div_times[td_ind].append(len(c) - 1 - j)
                    # a cell is picked to divide in the time step before it divides, i.e. 10 if divides in the 11th time
                    # interval
                    del td_ind
                del t_div

        num_cells[i] = len([1 for obj in c if obj.exists])  # number of currently existent cells
    obs = [num_cells, tvec]
    return c, obs


def discr_gen(par1):
    #  This discretized generation simulation will be used to test whether observed deviations from expected values
    #  in population growth simulations (with discretized time) arise as a result of differences in the distributions
    #  being sampled from in a population with "competitive" growth, or simply from the math being wrong. Note that
    #  no attention should be payed to the relative timing order of things here.
    num_gen = par1['num_gen']
    c = starting_popn(par1)  # gives us the starting population in the same way as with discretized time
    for i in range(num_gen):
        temp = len(c)
        for index in range(temp):
            if c[index].exists:
                c = next_gen(index, c, i+1, par1)
                # iterate through c to produce a new mother daughter pair for each cell
        del temp
    return c

# Analytical predictions


def vb_f(par1):
    l = np.log(2)/par1['td']
    if par1['modeltype'] in [1, 2]:  # seed cells with vb and wb around the analytically predicted means
        s_cd = par1['g2_std']*par1['td']
        cd = par1['CD']*par1['td']
        temp = par1['delta'] * np.exp(l*cd+0.5*(l*s_cd)**2)
    return temp


def wb_f(par1):
    l = np.log(2)/par1['td']
    if par1['modeltype'] in [1, 2]:  # seed cells with vb and wb around the analytically predicted means
        s_cd = par1['g2_std'] * par1['td']
        cd = par1['CD'] * par1['td']
        temp = par1['delta'] * (np.exp(l * cd + 0.5 * (l * s_cd) ** 2) - 1)
    return temp