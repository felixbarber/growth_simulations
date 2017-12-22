
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats


def do_the_time_warp(x, y, z):
    return np.array([x+y+z**4])

########################################################################################################################
# DISCRETIZED TIME SIMULATIONS
########################################################################################################################

# This notebook has symmetric division for yeast.


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
    delta = 2 ** (-CD / td)
    cellCount = 0  # total number of cells
    b = 1.0

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
        if par1['g1_thresh_std'] != 0:  # gaussian noise in the abundance of initiator required to cause initiation
            self.noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
        else:
            self.noise_thr = 0.0
        if par1['g1_std'] != 0:
            self.noise_g1 = np.random.normal(0.0, par1['g1_std'], 1)[0]
            # generate and retain the time additive noise in the first part of the cell cycle.
        else:
            self.noise_g1 = 0.0
        if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
            self.noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
        else:
            self.noise_g2 = 0.0
        if par1['initiator']:
            self.vi = ((self.delta + self.noise_thr - self.wb) / self.b + self.vb) * np.exp(
                self.noise_g1 * np.log(2) / self.td)
            self.vd = self.vi * np.exp((self.CD + self.noise_g2) * np.log(2) / self.td)
            self.wd = self.b * (self.vd - self.vi)
        elif not(par1['initiator']):
            self.vi = (self.wb + self.noise_thr) * np.exp(self.noise_g1 * np.log(2) / self.td)
            self.vd = self.vi * np.exp((self.CD + self.noise_g2) * np.log(2) / self.td)
            #self.wd = self.wb + self.K * (self.CD+self.noise_g2)
            self.wd = self.wb + self.K * self.CD
        self.t_grow = np.log(self.vd / self.vb) * self.td / np.log(2)
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


def next_gen(index, f, t, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    frac = 0.5

    f.append(Cell(frac*f[index].vd, frac*f[index].wd, t, mother=weakref.proxy(f[index])))
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell
    f[-1].isdaughter = True
    f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.
    f.append(Cell((1-frac)*f[index].vd, (1-frac)*f[index].wd, t, mother=weakref.proxy(f[index])))
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


def parameter_variation(par1, num_steps):
    tic = time.clock()
    g1_std = np.linspace(0.0, 0.10, 10)
    g2_std = np.linspace(0.0, 0.10, 10)
    dims = [len(g1_std), len(g2_std), 2]
    slopes = np.empty(dims)
    pcc = np.empty(dims)
    error = np.empty(dims)
    slopes_init = np.empty(dims)
    pcc_init = np.empty(dims)
    error_init = np.empty(dims)
    for k in range(dims[2]):
        par1['initiator'] = k
        par1['nstep'] = num_steps[k]
        for i in range(len(g1_std)):
            par1['g1_thresh_std'] = g1_std[-1 - i]
            #par1['g1_std'] = g1_std[-1 - i]
            for j in range(len(g2_std)):
                par1['g2_std'] = g2_std[j]
                c, num_cells, tvec = discr_time(par1)
                vb = [obj.vb for obj in c[1000:]]
                vd = [obj.vd for obj in c[1000:]]
                vi = [obj.vi for obj in c[1000:] if not(obj.nextgen is None)]
                vi_ng = [obj.nextgen.vi for obj in c[1000:] if not(obj.nextgen is None)]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, k] = vals[0]
                del vals
                vals = scipy.stats.pearsonr(vb, vd)
                pcc[i, j, k] = vals[0]
                error[i, j, k] = vals[1]
                del vals
                vals = scipy.stats.linregress(vi, vi_ng)
                slopes_init[i, j, k] = vals[0]
                del vals
                vals = scipy.stats.pearsonr(vi, vi_ng)
                pcc_init[i, j, k] = vals[0]
                error_init[i, j, k] = vals[1]
                del vals
            print('time taken: ', time.clock() - tic)
    return [slopes, pcc, error, slopes_init, pcc_init, error_init]


def heat_maps(obs, labels, g1_std, g2_std):
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    model = ['dilution symmetric', 'initiator symmetric']
    plots = [0, 1, 3, 4]
    for i in range(len(plots)):
        for j in range(obs[0].shape[2]):
            plt.figure(figsize=[11, 10])
            sns.heatmap(obs[plots[i]][:, :, j], xticklabels=np.around(g2_std, decimals=2), \
                             yticklabels=np.around(g1_std[::-1], decimals=2), annot=True)
            plt.xlabel('C+D timing noise $\sigma_{C+D}$',size=20)
            plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
            plt.title(labels[plots[i]]+' '+model[j], size=20)
            plt.show()


def test_initiation_times(c):
    t_init = [obj.td*np.log(obj.vi/obj.vb)/np.log(2) for obj in c]
    plt.figure(figsize=[6,6])
    sns.distplot(t_init)
    plt.title('Histogram of initiation times')
    plt.xlabel('initiation times')
    return t_init


def test_budding_times(c):
    t_vec = [obj.td*np.log(obj.vd/obj.vi)/np.log(2) for obj in c]
    plt.figure(figsize=[6,6])
    sns.distplot(t_vec)
    plt.title('Histogram of budding times')
    plt.xlabel('initiation times')
    return t_vec


def slope_vbvd_func(par1):
    cd = par['CD']
    td = par['td']
    l = np.log(2) / td
    s_cd = par1['g2_std']
    s_i = par1['g1_thresh_std']
    K = par['k']
    f = (s_cd / cd) ** 2 * (1 + 3 * l * cd) / ( \
            ((s_cd / cd) ** 2 + 3 + 3 * s_i ** 2 / (K * cd) ** 2) * np.exp((l * s_cd) ** 2) - 3)
    return f


def test_function(par1):
    g1_std = np.linspace(0.0, 0.1, 10)
    dims = g1_std.shape
    slopes = np.empty(dims)
    slopes_func = np.empty(dims)
    for i in range(len(g1_std)):
        par1['g1_thresh_std'] = g1_std[i]
        c, num_cells, tvec = discr_time(par1)
        vb = [obj.vb for obj in c[1000:]]
        vd = [obj.vd for obj in c[1000:]]
        vals = scipy.stats.linregress(vb, vd)
        slopes[i] = vals[0]
        slopes_func[i] = slope_vbvd_func(par1)
        del vals
    return slopes, slopes_func


def single_par_test(par1):
    # This function returns a list containing the slope between birth and division for a single input parameter
    c, num_cells, tvec = discr_time(par1)
    #x = np.asarray([obj.vi for obj in c[1000:] if not obj.daughter is None])
    x = np.asarray([obj.vb for obj in c[1000:]])
    #y = np.asarray([obj.daughter.vi for obj in c[1000:] if not obj.daughter is None])
    y = np.asarray([obj.vd for obj in c[1000:]])
    num_bins = 20
    l = 2.5
    fig=plt.figure(figsize=[7, 7])
    plt.hexbin(x, y, cmap="Purples", gridsize=30)
    tit = plt.title("Symmetric dilution Whi5 noiseless adder", size=16, weight="bold")
    plt.xlabel("$V_b$", size=16, weight="bold")
    plt.ylabel("$V_d$", size=16, weight="bold")
    plt.ylim(ymin=np.mean(y) - l * np.std(y), ymax=np.mean(y) + l * np.std(y))
    plt.xlim(xmin=np.mean(x) - l * np.std(x), xmax=np.mean(x) + l * np.std(x))
    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, np.arange(len(x)),
                                                                   statistic='mean', bins=num_bins, range=None)
    bin_posns = list([])
    y_av = list([])
    y_sem = list([])
    bin_err = list([])
    for j in range(num_bins):
        bin_no = binnumber == j + 1
        if np.sum(bin_no) > 20:  # we don't want to be skewed by having too few data points in a fit
            y_av.append(np.mean(y[np.nonzero(bin_no)]))
            y_sem.append(np.std(y[np.nonzero(bin_no)]) / np.sqrt(np.sum(bin_no)))
            # y_sem.append(np.std(y[np.nonzero(bin_no)]))
            bin_posns.append((bin_edges[j] + bin_edges[j + 1]) / 2)
            bin_err.append((bin_edges[j + 1] - bin_edges[j]) / 2)
    y_av = np.asarray(y_av)
    y_sem = np.asarray(y_sem)
    bin_posns = np.asarray(bin_posns)
    plt.errorbar(bin_posns, y_av, yerr=y_sem, label="binned means", ls="none", color="r")
    xvals = np.linspace(np.mean(x) - 2.5 * np.std(x), np.mean(x) + 2.5 * np.std(x), 2)
    # Generate values for plotting fits
    vals = stats.linregress(x, y)
    plt.plot(xvals, xvals * vals[0] + vals[1], 'b-', label="Regression slope ="+str(np.round(vals[0],2)))
    plt.legend()
    return vals[0],fig

