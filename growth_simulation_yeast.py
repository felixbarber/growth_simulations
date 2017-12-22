# Needs to have the following packages already imported:
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt


def do_the_time_warp(x,y,z):
    return np.array([x+y+z**4])

########################################################################################################################
# DISCRETIZED TIME SIMULATIONS
########################################################################################################################

# This notebook has asymmetric division of yeast and volumetric distribution of Whi5

# Now let's deal with this issue of discretized time simulations. First we define our classes for mother and daughter
# cells.

# This defines a global variable within growth_simulation which is necessary to define the Mother and Daughter classes.
# Anything that is defined in this variable is fixed for the course of the simulation. If you wish to iterate over a v
# variable you should include that as an input parameter for the function discr_time1
par = dict([('td', 1.0), ('num_s', 100), ('Vo', 1.0), ('Wo', 1.0), ('std_iv', 0.1),
            ('std_iw', 0.1), ('k', 1.0), ('std_it', 0.1)])
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
    CD = 0.5*td  # time delay C+D after the initiation is begun
    cellCount = 0  # total number of cells
    delta = 2**(-CD/td)

    def __init__(self, vb, w, tb, mother=None):
        # produces a new instance of the mother class. Only instance variables
        # initially are the initiator abundance at birth (w), the volume at birth,
        # the time of birth and it points to the mother.
        self.vb = vb
        self.wb = w
        self.tb = tb
        self.mother = mother  # references the mother cell
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.isdaughter = False  # boolean for whether the cell is a mother or daughter
        self.exists = True  # indexes whether the cell exists currently
        Cell.cellCount += 1

    def grow(self, par1):  # integration model for accumulation of initiator protein. In slow growth case.
        # Calculate the volume of this cell at initiation and division.
        self.R=par1['r']
        if par1['g1_thresh_std'] != 0:  # gaussian noise in the abundance of initiator required to cause initiation
            self.noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
        else:
            self.noise_thr = 0.0
        if par1['g1_std'] != 0:
            self.noise_g1 = np.random.normal(0.0, par1['g1_std'], 1)[0]
            # generate and retain the time additive noise in the first part of the cell cycle.
        else:
            self.noise_g1 = 0.0
        # note that noise is measured as a fraction of growth rate.
        self.vi = (self.wb+self.noise_thr)*np.exp(self.noise_g1*np.log(2)/self.td)
        if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
            self.noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
        else:
            self.noise_g2 = 0.0
        self.vd = self.vi * (1 + self.R) * np.exp(self.noise_g2*np.log(2)/self.td)
        self.wd = self.wb + self.K * np.log(self.vd/self.vi)*self.td/np.log(2)
        self.t_grow = np.maximum(np.log(self.vd / self.vb) * self.td / np.log(2), 0.0)
        self.t_div = self.tb + self.t_grow

# here par is a dictionary containing all the relevant parameters, a global variable defined in growth_simulation.


def starting_popn(par1):
    # here the boolean optional parameter 'dilution' specifies whether the model involved
    # should be based on a Whi5 dilution scheme.
    # par=dict([('dt', 1),('nstep',100), ('td', 90), ('num_s', 100),('Vo',1.0),('std_iv',0.1),('std_iw',0.1)])
    # Initialise simulation with a normal distribution of cell sizes and Whi5 abundances.
    v_init = np.random.normal(loc=par['Vo'], scale=par['std_iv'], size=par['num_s'])
    w_init = np.random.normal(loc=par['Wo'], scale=par['std_iv'], size=par['num_s'])
    # gives the fraction of the cell's doubling time
    # which this cell has passed through.
    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []
    t_div = np.random.uniform(low=0.0, high=1.0, size=par['num_s'])
    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        # Half are treated as daughters for the purposes of this simulation (since every odd cell has the previous even
        # cell as a daughter).
        c.append(Cell(v_init[i], w_init[i], 0))
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div-c[-1].t_grow
        # defined in this manner all starting cells have been born at time less than or equal to 0.
    del v_init, w_init
    return c


def next_gen(index, f, time, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    f.append(Cell(frac*f[index].vd, frac*f[index].wd, time, mother=weakref.proxy(f[index])))
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell
    f[-1].isdaughter = True
    f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.
    f.append(Cell((1-frac)*f[index].vd, (1-frac)*f[index].wd, time, mother=weakref.proxy(f[index])))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
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
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_popn(par1)
    num_cells[0] = len(c)
    for i in range(len(c)):
        if c[i].t_div < np.amax(tvec):
            td_ind = np.searchsorted(tvec, np.array(c[i].t_div), side='left', sorter=None)
            div_times[td_ind-1].append(i)  # a cell is picked to divide in the timestep before it divides, i.e. 10 if
            # divides in the 11th time interval.
            del td_ind
    # Now we begin iterating through the time values in tvec
    for i in range(nstep):
        for index in div_times[i]:
            c = next_gen(index, c, tvec[i + 1], par1)
            # set next gen growth-policy specific variables for mother and daughter cell.
            for j in range(2):
                t_div = c[len(c) - 1 - j].t_div
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the timeframe of our
                    # simulation
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


def single_cycle(par1, num_cells, dilution=True):
    v_init = 1.0  # starting volume for population of num_cells
    w_init = np.random.normal(loc=1.0, scale=0.2, size=num_cells)
    # W_init = np.ones(num_cells)  # starting Whi5 amount for population of num_cells
    f = []
    for i in range(num_cells):  # instantiating our initial population of cells. These cells do not have mothers.
        # Half are treated as daughters for the purposes of this simulation (since every odd cell has the previous even
        # cell as a daughter).
        f.append(Cell(v_init, w_init[i], 0))
        f[-1].grow(par1)
    # Now we generate new cells based on this single generation and look at the observed correlations
    c = []
    for index in range(len(f)):
        if dilution:
            # This function resets growth-policy specific variables for a single birth event in a whi5 dilution manner.
            # Should be used within discr_time to evolve the list of cells c.
            wd = f[index].wd*(f[index].vd-f[index].vs)/f[index].vd  # volumetric fraction of whi5 given
            c.append(Cell(f[index].vd - f[index].vs, wd, 0.0))
            # Produce a new cell based on the previous one and append it to the end of the list.
            c[-1].grow(par1)  # grow newborn cell
            c[-1].isdaughter = True  # track that this cell is a daughter
            # add new cell for newborn mother cell.
            c.append(Cell(f[index].vs, f[index].wd-wd, 0.0))
            c[-1].grow(par1)  # grow newborn cell
        else:
            # This function resets growth-policy specific variables for a single birth event in a whi5 indep manner.
            # Should be used within function discr_time to evolve the list of cells c.
            c.append(Cell(f[index].vd - f[index].vs, 0.0, 0.0))
            # Produce a new cell based on the previous one and append it to the end of the list.
            c[-1].grow(par1)  # grow newborn cell
            c[-1].isdaughter = True  # track that this cell is a daughter
            # add new cell for newborn mother cell.
            c.append(Cell(f[index].vs, 0.0, 0.0))
            c[-1].grow(par1)  # grow newborn cell
    del v_init, w_init
    return c
