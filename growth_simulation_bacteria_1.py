# Needs to have the following packages already imported:
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt


def do_the_time_warp(x,y,z):
    return np.array([x+y+z**4])

########################################################################################################################
# DISCRETIZED TIME SIMULATIONS -- fast bacterial growth
########################################################################################################################


# This defines a global variable within growth_simulation which is necessary to define the Mother and Daughter classes.
# Anything that is defined in this variable is fixed for the course of the simulation. If you wish to iterate over a v
# variable you should include that as an input parameter for the function discr_time1
par = dict([('td', 1.0), ('num_s', 100), ('Vo', 1.0), ('Wo', 1.0), ('std_iv', 0.1),
            ('std_iw', 0.1), ('std_it', 0.1)])
par['CD'] = 0.75*par['td']
par['k'] = 2**(1-par['CD']/par['td'])/par['CD']

# dt=time step
# nstep=number of time steps we go through
# td is the population doubling time
# num_s is the initial number of mothers and daughters
# Vo is the mean of the initial volume distribution.
# std_iv is the standard deviation of the starting volume distribution (same for both mothers and daughters)
# std_iw is the standard deviation of the starting whi5 distribution (same for both mothers and daughters)
# g1_std is the standard deviation in the passage time through g1
# g2_std is the standard deviation in the passage time through S/G2/M
# r is the asymmetry ratio in volume at division for mother and daughter cells


class Cell(object):
    # 'common base class for all mother cells of different growth policies'
    td = par['td']  # mass doubling time for cells
    K = par['k']   # production rate of initiator protein - produced in proportion to cell volume
    CD = par['CD']  # time delay C+D after the initiation is begun
    cellCount = 0  # total number of cells
    delta = 2 ** (-CD / td)
    b = 1.0

    def __init__(self, vb, w, tb, mother=None):
        # produces a new instance of the mother class. Only instance variables
        # initially are the initiator abundance at birth (w), the volume at birth,
        # the time of birth and it points to the mother.
        self.vb = vb
        self.wb = w
        self.tb = tb
        self.mother = mother  # references the mother cell
        self.daughter = []  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.exists = True  # indexes whether the cell exists currently
        self.vi = []
        self.num_init = 0
        self.died = False
        if mother is None:
            self.num_or = 1  # Keeps track of the number of origins of replication that cells have
            self.t_div = []  # Since this cell is now able to have multiple origins of replication exist simultaneously,
            # we are going to allow it to have multiple
        else:
            self.num_or = mother.num_or/2  # Has half the number of origins as the mother had (assuming equal division)
            self.t_div = self.mother.t_div[1:]  # remove the previous division event from that of the mother cell's list
            self.ti = self.mother.ti
        Cell.cellCount += 1

    def grow(self, par1):  # integration model for accumulation of initiator protein. In slow growth case.
        # Calculate the volume of this cell at initiation and division.
        if self.num_or < 1:
            self.died = True  # track that this cell didn't have enough origins of replication to keep going
            self.exists = False  # flag this cell so we don't propagate it

    def grow_init(self, par1):  # integration model for accumulation of initiator protein. In slow growth case.
        # Calculate the volume of this cell at initiation and division.
        if par1['g1_std'] != 0:
            self.noise_g1 = np.random.normal(0.0, par1['g1_std'], 1)[0]
            # generate and retain the time additive noise in the first part of the cell cycle.
        else:
            self.noise_g1 = 0.0

        vi = ((self.num_or * self.delta - self.wb) / self.b + self.vb) * np.exp(
            self.noise_g1 * np.log(2) / self.td)
        self.ti = self.td*np.log(vi/self.vb)/np.log(2)+self.tb  # The first initiation time

# here par is a dictionary containing all the relevant parameters, a global variable defined in growth_simulation.


def starting_popn(par1):
    #  Note that all starting cells have only one origin to begin with, so must go through initiation before they can
    #  divide.
    #  Initialise simulation with a normal distribution of cell sizes and initiator abundances.
    v_init = np.random.normal(loc=par['Vo'], scale=par['std_iv'], size=par['num_s'])
    w_init = np.random.normal(loc=par['Wo'], scale=par['std_iv'], size=par['num_s'])
    t_b = np.random.uniform(low=0.0, high=par['td'], size=par['num_s'])
    # gives the time of birth for this as a random variable between 0 and the doubling time of the population.
    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []
    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        # Cells have a randomly allocated initial abundance of the initiator protein.
        c.append(Cell(v_init[i], w_init[i], t_b[i]))
        c[-1].grow_init(par1)  # special growth policy for the first cell cycle, since this starts at birth.

        # At this stage all of our starting population has a time of initiation associated with it.
    del v_init, w_init, t_b
    return c


def initiation(index, f, time, par1):
    # Should be run before next_gen

    if np.abs(f[index].ti - time) > par1['dt']:
        raise ValueError('time allocation is wrong for initiations')
    # This function will pass one initiation for a single cell

    num_or = f[index].num_or
    f[index].num_or = num_or * 2  # doubles the number of origins
    f[index].vi = f[index].vb * np.exp(
        (f[index].ti - f[index].tb) * np.log(2) / f[index].td)  # update the volume at initiation to the current volume
    f[index].num_init += 1

    if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
        noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
    else:
        noise_g2 = 0.0

    f[index].t_div.append(f[index].CD + noise_g2 + f[index].ti)

    if par1['g1_std'] != 0:
        noise_g1 = np.random.normal(0.0, par1['g1_std'], 1)[0]
        # generate and retain the time additive noise in the first part of the cell cycle.
    else:
        noise_g1 = 0.0
    f[index].ti_old = f[index].ti
    f[index].ti = f[index].td * np.log(1 + f[index].num_or * f[index].delta / f[index].vi) / np.log(2) \
                  + f[index].ti + noise_g1

    # We have updated the number of origins, the volume at initiation, added a new division time for the previous
    # initiation, and added the time for the next initiation.
    return f


def next_gen(index, f, time, par1):
    if np.abs(f[index].t_div[0] - time) > par1['dt']:
        raise ValueError('time allocation is wrong for divisions')
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    frac = 0.5  # equal division of cellular resources between mother and daughter cell

    f.append(Cell(frac*f[index].vd, frac*f[index].wd, f[index].t_div[0], mother=weakref.proxy(f[index])))
    f[index].daughter.append(weakref.proxy(f[-1]))  # Update the mother cell to show this cell as a daughter.
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell

    # add new cell for newborn cell.
    f.append(Cell((1-frac)*f[index].vd, (1-frac)*f[index].wd, time, mother=weakref.proxy(f[index])))
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter.append(weakref.proxy(f[-1]))

    f[index].exists = False  # track that this cell no longer "exists".
    return f


def discr_time(par1):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a starting population of cells, and a set of parameters par
    # here par is a dictionary containing all the relevant parameters, a global variable defined in growth_simulation.
    # To begin we define our time vector
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    # Define lists which will keep track of the time step in which cells divide.
    div_times = []
    # Define list which will keep track of the time step in which cells initiate replication.
    init_times = []
    for i in range(nstep):
        div_times.append([])
        init_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_popn(par1)  # generating our starting population
    num_cells[0] = len(c)  # keeps track of how many cells we had at each timestep
    for i in range(len(c)):
        # Now we track when each cell in our starting population will go through initiation
        if c[i].ti < np.amax(tvec):
            ti_ind = np.searchsorted(tvec, np.array(c[i].ti), side='left', sorter=None)
            init_times[ti_ind-1].append(i)  # a cell is picked to initiate in the timestep before it initiates
            # , i.e. 10 if divides in the 11th time interval.
            del ti_ind

    # Now we begin iterating through the time values in tvec
    for i in range(nstep):  # We start our time vector at one timestep in. Note tvec starts at 0 and has nstep+1.
        # First we go through the initiations. It's important that this happen first because this way cells which
        # have their new division time greater than or equal to the current timestep will divide.
        for index in init_times[i]:
            c = initiation(index, c, tvec[i + 1], par1)
            if len(c[index].t_div) == 1:
                #  If the division event that was just added from this initiation is the first one recorded for this
                # cell (we expect this to be true for slow growth), we mark it for division based on this division event
                # Note that if this is not the case then we expect that this cell will have already been marked for
                # division from either it's birth event or from it's previous initiation.
                t_div = c[index].t_div[0]
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the timeframe of our
                    # simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    div_times[td_ind - 1].append(index)  # daughters
                    del td_ind
                del t_div
            if c[index].ti <= c[index].t_div[0]:
                # If this cell is going to go through another round of initiation before the division mark it as such
                t_init = c[index].ti
                if t_init < np.amax(tvec):  # We only mark cells for division if they fall within the timeframe of our
                    # simulation
                    ti_ind = np.searchsorted(tvec, np.array(c[index].ti), side='left', sorter=None)
                    init_times[ti_ind - 1].append(index)
                    del ti_ind
                del t_init
        # Now we go through the cells which have been marked for division, and update their progeny accordingly.
        for index in div_times[i]:
            # Now we need to update the values at division for the cells which are about to divide.
            if c[index].num_init > 0:  # In this case we know we can calculate based on the output of initiator
                c[index].vd = c[index].vi * np.exp((c[index].t_div[0] - c[index].ti_old) * np.log(2) / c[index].td)
                c[index].wd = par['k'] * (c[index].vd - c[index].vi)
            else:
                c[index].vd = c[index].vb * np.exp(c[index].t_div[0])
                c[index].wd = c[index].wb + par['k']*(c[index].vd-c[index].vb)
            c = next_gen(index, c, tvec[i + 1], par1)  # Newborn cells now have inherited all the division times from
            # their mother, as well as their mother's initiation time.
            # If the initiation time for these cells is either less than the first listed division time or the cell has
            # no listed division times (expected for slow growth) mark the two newborn cells for initiation.
            if len(c[-1].t_div) == 0 or c[-1].t_div[0] >= c[-1].ti:
                for j in range(2):
                    c[len(c)-1-j].vi = c[len(c)-1-j].vb  # This is a placeholder so that the
                    t_init = c[len(c) - 1 - j].ti
                    if t_init < np.amax(tvec):
                        # We only mark cells for initiation if they fall within the timeframe of our simulation
                        ti_ind = np.searchsorted(tvec, np.array(t_init), side='left', sorter=None)
                        init_times[ti_ind - 1].append(len(c) - 1 - j)  # daughters
                        del ti_ind
                    del t_init
            # If the initiation time for these new cells is greater than the division time we mark them for division
            else:
                for j in range(2):
                    c[-1-j].vd = c[-1-j].vb * np.exp(c[-1-j].t_div[0] * np.log(2) / c[index].td)
                    c[-1-j].wd = par['k'] * (c[-1-j].vd - c[-1-j].vb)
                if t_div < np.amax(tvec):
                    # We only mark cells for division if they fall within the timeframe of our simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    div_times[td_ind - 1].append(len(c) - 1 - j)  # daughters
                    del td_ind
                del t_div
        num_cells[i+1] = len(c)
        # if np.mod(i,100) == 0:
        #        print('Time step: ', i)
    return c, num_cells, tvec


def simulation_stats(d, m):
    std_d = np.std(d[0, :])
    mean_d = np.mean(d[0, :])
    cv_d = scipy.stats.variation(d[0, :])
    std_m = np.std(m[0, :])
    mean_m = np.mean(m[0, :])
    cv_m = scipy.stats.variation(m[0, :])
    std_vec = np.array([std_d, std_m])
    mean_vec = np.array([mean_d, mean_m])
    cv_vec = np.array([cv_d, cv_m])
    return std_vec, mean_vec, cv_vec


def dplot(x, y, st):
    vals = scipy.stats.linregress(x,y)
    xmax = np.mean(x) + 2.5 * np.std(x)
    xmin = np.mean(x) - 2.5 * np.std(x)
    ymin = np.mean(y) - 2.5 * np.std(y)
    ymax = np.mean(y) + 2.5 * np.std(y)
    xv = np.array([xmin, xmax])
    plt.figure(figsize=[10, 10])
    plt.hexbin(x, y, cmap="Purples", gridsize=50)
    plt.plot(xv, vals[0] * xv + vals[1], label='slope= ' + '%.3f' % (vals[0]) + '$\pm$''%.3f' % (vals[4]))
    plt.title(st, size=30, y=1.04)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.legend(fontsize='x-large')
    plt.xlabel('$V_b$', size=30, weight='bold')
    plt.ylabel('$V_d$', size=30, weight='bold')
    plt.show()
    return vals


def single_cycle(par1, num_cells):
    v_init = 1.0  # starting volume for population of num_cells
    w_init = 1.0
    # W_init = np.ones(num_cells)  # starting Whi5 amount for population of num_cells
    # Now we generate new cells based on this single generation and look at the observed correlations
    c = []
    wd = w_init + 2**0.25
    vd = w_init * 2**0.75
    for index in range(num_cells):
        # This function resets growth-policy specific variables for a single birth event in a whi5 dilution manner.
        # Should be used within discr_time to evolve the list of cells c.
        wb = wd*0.5  # volumetric fraction of whi5 given
        c.append(Cell(vd*0.5, wb, 0.0))
        # Produce a new cell based on the previous one and append it to the end of the list.
        c[-1].grow(par1)  # grow newborn cell
        c[-1].isdaughter = True  # track that this cell is a daughter
        # add new cell for newborn mother cell.
        c.append(Cell(vd*0.5, wb, 0.0))
        c[-1].grow(par1)  # grow newborn cell
    del v_init, w_init
    return c
