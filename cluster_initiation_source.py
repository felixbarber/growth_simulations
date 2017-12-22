import numpy as np
import scipy
import weakref
import time
from scipy import stats


def test(x, y, z):
    return np.array([x+y+z**3])


########################################################################################################################
# DISCRETIZED TIME SIMULATIONS
########################################################################################################################

# This notebook has asymmetric division for yeast.


# This defines a global variable within growth_simulation which is necessary to define the Mother and Daughter classes.
# Anything that is defined in this variable is fixed for the course of the simulation. If you wish to iterate over a v
# variable you should include that as an input parameter for the function discr_time1
par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('wd', 1.0), ('wm', 1.0), ('std_v', 0.2), ('std_w', 0.2)])
models = ['init accum noise: gr, start, asymm','Vb min, init accum noise: gr, start, asymm']
list1 = [1, 2]  # models which include noise in the growth rate
list2 = [1, 2, 3, 4]  # models with additive noise in passage through Start
list3 = [1, 2, 3, 4]  # models with noise in G2 going exclusively to the daughter cell
list4 = [2, 4]     # models with a minimum volume at which cells go through Start (no negative growth)
list5 = [3, 4]     # models with noise just in r rather than in budded phase duration
list6 = [1, 2, 3, 4]  # models with a volumetric fraction of initiator going to both mother and daughter cells.


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
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.isdaughter = False  # boolean for whether the cell is a mother or daughter
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

        if par1['modeltype'] in list1 and not par1['l_std'] == 0:
            l = (1 + np.random.normal(0.0, par1['l_std'], 1)[0]) * np.log(2) / par1['td']
            # allows for having noise in growth rate.

            # prevents shrinking due to a negative growth rate
            while l < 0:  # never allow a negative growth rate as this implies cells are shrinking
                l = (1 + np.random.normal(0.0, par1['l_std'], 1)[0]) * np.log(2) / par1[
                    'td']

        else:
            l = np.log(2)/par1['td']

        # G1 Part of the cell cycle

        if par1['modeltype'] in list2 and par1['g1_thresh_std'] != 0:
            noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
        else:
            noise_thr = 0.0
        if par1['modeltype'] in list4:
            self.vi = max(self.vb - self.wb + par1['delta']*(1+noise_thr), self.vb)
        else:
            self.vi = self.vb - self.wb + par1['delta']*(1+noise_thr)
        del noise_thr

        # G2 part of the cell cycle

        self.vd = self.vi - 1  # ensures that the following code is run at least once.

        # if par1['modeltype'] in list5:
        #     if par1['r_std'] != 0:
        #         self.vd = self.vi * (1 + par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 1)[0]))
        #     else:
        #         self.vd = self.vi * (1 + par1['r'])
        # else:
        #     if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
        #         noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
        #     else:
        #         noise_g2 = 0.0
        #     self.vd = self.vi * np.exp((par1['CD'] + noise_g2) * par1['td'] * l)

        # prevents shrinking in cells in list8. If we do this for all models then you're trying to make a positive * neg
        # = a positive and it takes forever.
        if par1['modeltype'] in list4:
            while self.vd <= self.vi:  # resample until we get a volume at division which is greater than that at birth.
                if par1['modeltype'] in list5:
                    if par1['r_std']!=0:
                        self.vd = self.vi * (1 + par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 1)[0]))
                    else:
                        self.vd = self.vi * (1 + par1['r'])
                else:
                    if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                        noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
                    else:
                        noise_g2 = 0.0
                    self.vd = self.vi * np.exp((par1['CD'] + noise_g2) * par1['td'] * l)
        else:
            if par1['modeltype'] in list5:
                if par1['r_std'] != 0:
                    self.vd = self.vi * (1 + par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 1)[0]))
                else:
                    self.vd = self.vi * (1 + par1['r'])
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

        if self.mother:
            if self.isdaughter:  # if this cell has a mother we update the mother values to account for the new gen
                # print self.mother.gen_d
                # print self.mother.gen_m
                self.gen_d = self.mother.gen_d + 1
            elif not self.isdaughter:
                # print self.mother.gen_d
                # print self.mother.gen_m
                self.gen_m = self.mother.gen_m + 1
        Cell.cellCount += 1


def starting_popn(par1):
    # To clarify we first set the initial condition for the simulation.
    l = np.log(2)/par1['td']
    if par1['modeltype'] in [1, 2]:
        s_l = par1['l_std']*l
        s_cd = par1['g2_std']*par1['td']
        cd = par1['CD']*par1['td']
        wd = 2 * par1['delta'] * (exp_xy(l, s_l, cd, s_cd)+exp_xy(-l, s_l, cd, s_cd)-2)
        wm = 2 * par1['delta'] * (1-exp_xy(-l, s_l, cd, s_cd))
        vd = 2 * par1['delta'] * (1-exp_xy(-l, s_l, cd, s_cd))
        vm = 2 * par1['delta']
    if par1['modeltype'] in [3, 4]:
        if par1['r_std'] == 0:
            r = par1['r']
        else:
            r = par1['r'] * (1 + np.random.normal(0, par1['r_std'], 10 ** 4))
        wd = 2 * par1['delta'] * np.mean(r**2/(1+r))
        wm = 2 * par1['delta'] * np.mean(r/(1+r))
        vd = 2 * par1['r'] * par1['delta']
        vm = 2 * par1['delta']
        # here the boolean optional parameter 'dilution' specifies whether the model involved
    # should be based on a Whi5 dilution scheme.
    # par=dict([('dt', 1),('nstep',100), ('td', 90), ('num_s', 100),('Vo',1.0),('std_iv',0.1),('std_iw',0.1)])
    # Initialise simulation with a normal distribution of cell sizes and Whi5 abundances.
    v_init_d = np.random.normal(loc=vd, scale=par['std_v']*vd, size=par['num_s'])
    v_init_m = np.random.normal(loc=vm, scale=par['std_v']*vm, size=par['num_s'])
    w_init_d = np.random.normal(loc=wd, scale=par['std_w']*wd, size=par['num_s'])
    w_init_m = np.random.normal(loc=wm, scale=par['std_w']*wm, size=par['num_s'])
    t_div = np.random.uniform(low=0.0, high=1.0, size=par['num_s'])
    # gives the fraction of the cell's doubling time
    # which this cell has passed through.
    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []

    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        # Daughters
        c.append(Cell(v_init_d[i], w_init_d[i], 0))
        c[-1].isdaughter = True
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div-c[-1].t_grow

        # Mothers
        c.append(Cell(v_init_m[i], w_init_m[i], 0))
        c[-1].isdaughter = False
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        # defined in this manner all starting cells have been born at time less than or equal to 0.
    del v_init_d, v_init_m, w_init_d, w_init_m
    return c


def starting_popn_seeded(c, par1):
    # c is the set of "existent" cells at the end of the previous simulation, done so as to yield a converged
    # distribution of cells

    indices = np.random.randint(low=0, high=len(c), size=par1['num_s1'])
    temp = [c[ind] for ind in indices]
    val = []
    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(Cell(obj.vb, obj.wb, 0.0, mother=obj.mother))
        val[-1].isdaughter = obj.isdaughter
        val[-1].grow(par1)  # randomly generate the final state of this seeding cell.

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
    if par1['modeltype'] in list6:
        frac1 = (f[index].vd-f[index].vi)/f[index].vd
    f.append(Cell(f[index].vd - f[index].vi, frac1*f[index].wd, t, mother=weakref.proxy(f[index])))  # generates
    # daughter cell.
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].isdaughter = True
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.
    f.append(Cell(f[index].vi, (1-frac1)*f[index].wd, t, mother=weakref.proxy(f[index])))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    return f


def discr_time(par1):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * par1['td'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep+1):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_popn(par1)
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
    for i in range(nstep+1):
        if i > 0:
            num_div_d[i] = num_div_d[i-1]
            num_div_m[i] = num_div_m[i-1]
        for index in div_times[i]:  # the presence of indices in the 0th timestep means that the seeding cells grew
            # negatively. Statistically unlikely but may happen for one or two cells.

            # Double check that these indices have been allocated correctly: c[index].t_div should be in the interval
            # (tvec[i-1], tvec[i]]
            if i != 0:
                if tvec[i-1] >= c[index].t_div or tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i+1]
                    raise ValueError('The cell division orders are wrong')
            elif i == 0:
                if tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i+1]
                    raise ValueError('The cell division orders are wrong')

            c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
            if c[index].isdaughter:
                num_div_d[i] += 1
            elif not c[index].isdaughter:
                num_div_m[i] += 1
            for j in range(2):  # Set the time of division for the two new cells produced by next_gen
                t_div = c[len(c) - 1 - j].t_div
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the time frame of our
                    # simulation
                    c[len(c) - 1 - j].should_div = True  # this cell should for sure divide at some stage within the sim
                    # simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    if td_ind < i:
                        print "Timestep", i, "div timestep", td_ind, "t_div", t_div, "t_birth", c[len(c)-1-j].tb, "t_div_prev", c[index].t_div
                        raise ValueError('Cells are falling behind')
                    div_times[td_ind].append(len(c) - 1 - j)
                    # a cell is picked to divide in the time step before it divides, i.e. 10 if divides in the 11th time
                    # interval
                    del td_ind
                del t_div
        num_cells[i] = len(c)
    obs = [num_cells, tvec, num_div_d, num_div_m]
    return c, obs


def discr_time_1(par1, starting_pop):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1 and a starting population of cells.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * par1['td'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
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
        if i > 0:
            num_div_d[i] = num_div_d[i - 1]
            num_div_m[i] = num_div_m[i - 1]
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
            if c[index].isdaughter:
                num_div_d[i] += 1
            elif not c[index].isdaughter:
                num_div_m[i] += 1
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
        num_cells[i] = len(c)
    obs = [num_cells, tvec, num_div_d, num_div_m]
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


def discr_gen_1(par1, starting_pop):  # takes a starting population of cells as input.
    #  This discretized generation simulation will be used to test whether observed deviations from expected values
    #  in population growth simulations (with discretized time) arise as a result of differences in the distributions
    #  being sampled from in a population with "competitive" growth, or simply from the math being wrong. Note that
    #  no attention should be payed to the relative timing order of things here.
    num_gen = par1['num_gen1']  # note that this allows this function to be called independently of discr_gen.
    c = starting_pop[:]  # gives us the starting population in the same way as with discretized time
    for i in range(num_gen):
        r = len(c)
        for index in range(r):
            if c[index].exists:
                c = next_gen(index, c, i+1, par1)
                # iterate through c to produce a new mother daughter pair for each cell
    return c

########################################################################################################################
# SINGLE PARAMETER SET MEASUREMENT CODE
########################################################################################################################


def single_par_meas(par1):
    c, temp = discr_time(par1)
    obs = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0,k] = vals[0]
        del vals
        obs[1,k] = np.mean(vb)
        obs[2,k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3,k] = np.mean(vd)
        obs[4,k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5,k] = np.mean(wb)
        obs[6,k] = num_neg
        obs[7,k] = num_cell
    return obs


def single_par_meas2(par1):
    c = discr_gen(par1)
    obs = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        num_lc = len([obj for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0])*100.0/len(vb)
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
        obs[6, k] = np.mean(np.asarray(wb)*np.asarray(wb))
        obs[7, k] = num_lc
    del c
    return obs


def single_par_meas3(par1):
    c, num_cells, tvec, av_whi5, num_div_d, num_div_m = discr_time(par1)
    obs = np.empty([8, 2])
    obs1 = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0,k] = vals[0]
        del vals
        obs[1,k] = np.mean(vb)
        obs[2,k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3,k] = np.mean(vd)
        obs[4,k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5,k] = np.mean(wb)
        obs[6,k] = num_neg
        obs[7,k] = num_cell

        vb = [obj.vb for obj in c[1000:] if
              obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs1[0, k] = vals[0]
        del vals
        obs1[1, k] = np.mean(vb)
        obs1[2, k] = np.mean(np.asarray(vb) * np.asarray(vb))
        obs1[3, k] = np.mean(vd)
        obs1[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs1[5, k] = np.mean(wb)
        obs1[6, k] = num_neg
        obs1[7, k] = num_cell
    return obs, obs1


def single_par_meas4(par1):  # same as single_par_meas2 except that it also returns the distribution of growth times
    c = discr_gen(par1)
    obs = np.empty([6, 2])
    t_grow = []
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        t_grow.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k]))
        # mothers first, then daughters
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
    del c
    return obs, t_grow


def single_par_meas5(par1):  # same as single_par_meas3 except that it also returns the distribution of growth times
    c, num_cells, tvec, av_whi5, num_div_d, num_div_m = discr_time(par1)
    obs = np.empty([8, 2])
    obs1 = np.empty([8, 2])
    t_grow = []
    t_grow1 = []
    if len(c) < 1000:
        raise ValueError('error in parameter set CD='+str(np.round(par1['CD'], 2))+' g1_std='
                         + str(np.round(par1['g1_thresh_std'], 2))+' g2_std'+str(np.round(par1['g2_std'], 2)))
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        t_grow.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k and obj.exists]))

        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
        obs[6, k] = num_neg
        obs[7, k] = num_cell

        vb = [obj.vb for obj in c[1000:] if
              obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        t_grow1.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k]))
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs1[0, k] = vals[0]
        del vals
        obs1[1, k] = np.mean(vb)
        obs1[2, k] = np.mean(np.asarray(vb) * np.asarray(vb))
        obs1[3, k] = np.mean(vd)
        obs1[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs1[5, k] = np.mean(wb)
        obs1[6, k] = num_neg
        obs1[7, k] = num_cell
    return obs, obs1, t_grow, t_grow1


def single_par_meas6(par1):  # same as single_par_meas2 except that it also returns the distribution of growth times
    c = discr_gen(par1)
    obs = np.empty([6, 3])
    for k in range(3):
        if k < 2:
            vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
            vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
            wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        elif k == 2:
            vb = [obj.vb for obj in c[1000:]]  # mothers are at 0, then daughters are at 1.
            vd = [obj.vd for obj in c[1000:]]
            wb = [obj.wb for obj in c[1000:]]
        # mothers first, then daughters
        val = scipy.stats.linregress(vb, vd)
        obs[0, k] = val[0]
        del val
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
    del c
    return obs  # returns mothers, daughters, and whole population respectively.


########################################################################################################################
# This is to test the functional form of the correlation between Vb and Vd for daughter cells with asymmetric growth
# and no noise in the production of Whi5. Works with obs being the output of parameter_variation_asymmetric
########################################################################################################################
def exp_xy(l, s_l, cd, s_cd):
    exy = np.exp(0.5*((l*s_cd)**2+(cd*s_l)**2+2*l*cd)/(1-s_l**2*s_cd**2))/np.sqrt(1-s_l**2*s_cd**2)
    return exy


def y_exp_xy(l, s_l, cd, s_cd):
    val = np.exp((2 * cd * l + l ** 2 * s_cd ** 2 + cd ** 2 * s_l ** 2) / (2 - 2 * s_cd ** 2 * s_l ** 2)) * (
        cd + l * s_cd ** 2) / (1 - s_cd ** 2 * s_l ** 2) ** (3 / 2)
    return val


def yy_exp_xy(l, s_l, cd, s_cd):
    val = np.exp((2 * cd * l + l ** 2 * s_cd ** 2 + cd ** 2 * s_l ** 2) / (2 - 2 * s_cd ** 2 * s_l ** 2)) * (s_cd ** 2
            + (cd + l * s_cd ** 2) ** 2 - s_cd ** 4 * s_l ** 2) / (1 - s_cd ** 2*s_l ** 2) ** (5 / 2)
    return val


def tay_e1(s1):
    return 1 + s1**2 + s1**4 + s1**6 + s1**8


def tay_e2(s1):
    return 1 + 3*s1**2 + 5*s1**4 + 7*s1**6 + 9*s1**8


def wbwb(par1, celltype):  # units should be volume and time for s_i and s_cd respectively.
    if par1['modeltype'] in [3, 4]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 2:
            val = 0.5*vivi(par1, celltype=2)*np.mean(temp**2*(1+temp**2)/(1+temp)**2)
    return val


def vbvb(par1, celltype):
    if par1['modeltype'] in [3, 4]:
        if celltype == 0:  # mothers
            val = vivi(par1, celltype=2)
        if celltype == 1:  # daughters
            val = par1['r']**2*(1+par1['r_std']**2)*vivi(par1, celltype=2)
        if celltype == 2:  # population
            val = 0.5*vivi(par1, celltype=2)*(par1['r']**2*(1+par1['r_std']**2)+1)
    return val


def vivi(par1, celltype):
    if par1['modeltype'] in [3, 4]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 2:
            val = par1['delta']**2*(3+par1['g1_thresh_std']**2)/(1-0.5*np.mean((1+temp**2)/(1+temp)**2))
    return val


def vb(par1, celltype):
    if par1['modeltype'] in [3, 4]:
        if celltype == 0:  # mothers
            val = 2*par1['delta']
        elif celltype == 1:  # daughters
            val = 2 * par1['r'] * par1['delta']
        elif celltype == 2:  # population
            val = par1['delta'] * (1+par1['r'])
    return val


def vd(par1, celltype):
    if par1['modeltype'] in [3, 4]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 0:  # mothers
            val = par1['delta'] * (2 * np.mean(1.0 / (1 + temp)) + 1) * (1 + par1['r'])
        elif celltype == 1:  # daughters
            val = par1['delta'] * (2 * np.mean(temp / (1 + temp)) + 1) * (1 + par1['r'])
        elif celltype == 2:  # population
            val = 2 * par1['delta'] * (1+par1['r'])
    return val


def vdvb(par1, celltype):
    if par1['modeltype'] in [3, 4]:
        temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10 ** 5))
        if celltype == 0:
            val = (2*par1['delta']**2+vivi(par1, celltype=2)*np.mean(1.0/(1+temp))) * (1+par1['r'])
        elif celltype == 1:
            val = (1+par1['r'])*(2*par1['delta']**2*par1['r']+vivi(par1, celltype=2)*np.mean(temp**2/(1+temp)))
        elif celltype == 2:
            val = 0.5*((2*par1['delta']**2+vivi(par1, celltype=2)*np.mean(1.0/(1+temp))) * (1+par1['r'])+(1+par1['r'])*
                       (2*par1['delta']**2*par1['r']+vivi(par1, celltype=2)*np.mean(temp**2/(1+temp))))
    return val


def slope_vbvd(par1, celltype):
    val = (vdvb(par1, celltype) - vd(par1, celltype) * vb(par1, celltype)) /\
        (vbvb(par1, celltype) - vb(par1, celltype) ** 2)
    return val

########################################################################################################################


def func(par1):
    tic = time.clock()
    g1_std = np.linspace(0.0, 0.10, 10)
    g2_std = np.linspace(0.0, 0.10, 10)
    cd = np.linspace(0.5, 1.0, 11)
    dims = [len(g1_std), len(g2_std), 2, len(cd)]
    slopes = np.empty(dims)
    par1['initiator'] = 0  # dilution model
    for k in range(len(cd)):
        par1['CD'] = cd[k]
        for i in range(len(g1_std)):
            par1['g1_thresh_std'] = g1_std[-1 - i]
            for j in range(len(g2_std)):
                par1['g2_std'] = g2_std[j]
                c, num_cells, tvec = discr_time(par1)
                vb = [obj.vb for obj in c[1000:] if obj.isdaughter]
                vd = [obj.vd for obj in c[1000:] if obj.isdaughter]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, 0, k] = vals[0]
                del vals, vb, vd
                vb = [obj.vb for obj in c[1000:] if not obj.isdaughter]
                vd = [obj.vd for obj in c[1000:] if not obj.isdaughter]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, 1, k] = vals[0]
        print('time taken: ', time.clock() - tic)
    return slopes


def parameter_variation_asymmetric(par1):  # here we only look at the dilution growth policy, and are
    # saving the result for both mothers and daughters
    tic = time.clock()
    g1_std = np.linspace(0.0, 0.20, 21)
    g2_std = np.linspace(0.0, 0.20, 21)
    dims = [len(g1_std), len(g2_std), 2]
    slopes = np.empty(dims)
    pcc = np.empty(dims)
    error = np.empty(dims)
    var = np.empty([len(g1_std), len(g2_std), 2, 4])
    for i in range(len(g1_std)):
        par1['g1_thresh_std'] = g1_std[-1 - i]
        for j in range(len(g2_std)):
            par1['g2_std'] = g2_std[j]
            c, num_cells, tvec = discr_time(par1)
            for k in range(2):
                vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
                vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, k] = vals[0]
                del vals
                vals = scipy.stats.pearsonr(vb, vd)
                pcc[i, j, k] = vals[0]
                error[i, j, k] = vals[1]
                del vals
                var[i, j, k, 0] = np.mean(vb)
                var[i, j, k, 1] = np.std(vb)
                var[i, j, k, 2] = np.mean(vd)
                var[i, j, k, 3] = np.mean(np.asarray(vd)*np.asarray(vb))
        print('time taken: ', time.clock() - tic)
    obs = [slopes, pcc, error, var]
    return obs


def exp_xy(x, s_x, y, s_y):
    exy = np.exp(0.5*((x*s_y)**2+(y*s_x)**2+2*x*y)/(1-s_x**2*s_y**2))/np.sqrt(1-s_x**2*s_y**2)
    return exy


def starting_popn_seeded_1(c, par1, discr_time=False):
    # c is the set of "existent" cells at the end of the previous simulation, done so as to yield a converged
    # distribution of cells

    indices = np.random.randint(low=0, high=len(c), size=par1['num_s1'])
    temp = [c[ind] for ind in indices]
    val = []

    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(Cell(obj.vb, obj.wb, 0.0, mother=obj.mother))  # note that depending on how this code is implemented,
        # obj.mother may not exist. We therefore do the following.
        # reset all the variables to be the same as that in the previous cell, so we don't have to 'grow' them again.
        val[-1].vi = obj.vi
        val[-1].wd = obj.wd
        val[-1].vd = obj.vd
        val[-1].isdaughter = obj.isdaughter
        val[-1].t_grow = obj.t_grow
        val[-1].gen_d = obj.gen_d
        val[-1].gen_m = obj.gen_m

    if discr_time:
        nstep = par1['nstep']
        t_max = nstep * par1['dt'] * par1['td']  # maximum time of the previous simulation, so that we know what time to
        # start our new cells at. Only use this if the previous simulation was a discretized time one.
        # Otherwise start at random.
        for i0 in range(len(val)):
            val[i0].t_div = temp[i0].t_div - t_max
            val[i0].tb = val[i0].t_div - val[i0].t_grow
    else:
        t_div = np.random.uniform(low=0.0, high=1.0, size=par1['num_s1'])
        # gives the fraction of the cell's doubling time which this cell has passed through when we start tracking it.
        # Now we start a list which will keep track of all currently existing cells.
        # We will have the same list for mothers
        # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
        for i0 in range(par1['num_s1']):
            val[i0].t_div = max(t_div[i0] * val[i0].t_grow, par1['dt']*par1['td']*1.01)  # to ensure that this cell gets
            # detected in the iteration through calculating next generation cells.
            # we expect that these cells have been caught at
            # some uniformly distributed point of progression through their cell cycles.
            val[i0].tb = val[i0].t_div-val[i0].t_grow

    output = np.zeros([11, len(val)])  # return also all variables to reproduce this starting population.
    for i0 in range(len(val)):
        output[0, i0] = val[i0].vb
        output[1, i0] = val[i0].wb
        output[2, i0] = val[i0].tb
        output[3, i0] = val[i0].vi
        output[4, i0] = val[i0].wd
        output[5, i0] = val[i0].vd
        output[6, i0] = val[i0].isdaughter
        output[7, i0] = val[i0].t_grow
        output[8, i0] = val[i0].gen_d
        output[9, i0] = val[i0].gen_m
        output[10, i0] = val[i0].t_div

    for i0 in range(len(val)):
        if val[i0].t_div < 0:
            raise ValueError('Error in starting_popn_seeded')
    return val, output
