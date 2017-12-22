# Needs to have the following packages already imported:
import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt

def do_the_time_warp(x,y,z):
    return np.array([x+y+z**3])

# This function takes as inputs 2XN np arrays d and m which
# contain the volume and Whi5 distributions for cells [[Vol],[Whi5]], and
# a list par. This function can in principle take on different
# numbers of daughters and mothers within the population.
#def incr_mothers_simulator(d,m,par):
def incr_mothers_simulator(d,m,par):
    #par =list([td,num_gen,std_g1,std_bud,f,delta,k (production rate of Whi5),r])
    for i in range(par[1]):
        if not(par[2]==0):
            noise_g1=np.random.normal(0,par[2],d.shape[1]+m.shape[1]) # Noise for both mother and daughter populations
        else: noise_g1=np.zeros(d.shape[1]+m.shape[1])
        if not(par[3] == 0):
            noise_bud = np.random.normal(0, par[3], d.shape[1] + m.shape[1]) # Noise for both mother and daughter populations
        else: noise_bud=np.zeros(d.shape[1]+m.shape[1])
            # update the abundance of Whi5 for the next generation, assuming noise in passage through Start
        d_temp = np.empty([d.shape[0],m.shape[1]+d.shape[1]]) # temporary array to store the whi5 and volume measurements.
        m_temp = np.empty([m.shape[0],
                       m.shape[1] + d.shape[1]])  # temporary array to store the whi5 and volume measurements.
        # store the Whi5 data for daughters born from mother cells first
        d_temp[1,:m.shape[1]] = par[4] * (m[1, :] + par[5] + noise_bud[:m.shape[1]] * par[6])
        # Now store the values for daughters born from daughters
        d_temp[1,m.shape[1]:] =par[4] * (d[1, :] + par[5] + noise_bud[m.shape[1]:d.shape[1] + m.shape[1]] * par[6])
        # Mother amount of Whi5 related to daughter amount by (1-f)/f. The fact that we have noise in budded phase
        # shouldn't affect this
        m_temp[1,:] = (1 - par[4]) / par[4] * d_temp[1,:]
        # store the volumes for the mothers born from mothers first. Update directly from the amount of Whi5 the cell has
        # since Whi5 is measured in units of volume here. Have to do from previous generation because we look at Vb
        # amount of Whi5 is deterministic in setting timing of G1, and then add noise on top.
        m_temp[0,:m.shape[1]] = np.max(m, 0) * np.exp(noise_g1[:m.shape[1]] * np.log(2) / par[0])
        # mothers from daughters
        m_temp[0,m.shape[1]:] = np.max(d, 0) * np.exp(
            noise_g1[m.shape[1]:d.shape[1] + m.shape[1]] * np.log(2) / par[0])
        # We use mother volume to also define the volume of the next gen. daughter. Vs(2^(tn/Td)*(1+r)-1)
        d_temp[0,:] = m_temp[0,:] * (np.exp(noise_bud[:] * np.log(2) / par[0]) * (1 + par[7]) - 1)
        del m,d
        m=np.empty(m_temp.shape)
        d=np.empty(d_temp.shape)
        m[:,:]=m_temp[:,:]
        d[:,:]=d_temp[:,:]
        del m_temp,d_temp
    return d,m

def incr_mothers_simulator1(d,m,par): # this differs from the above incr_mothers_simulator in that it includes a period T2
    # after initiating Start but before budding occurs. We assume that T2 is the same for daughters and mothers.
    # This is the same as for incr_mothers_simulator except that we shorten the period prior to the budded phase. The noise
    # in g1 is added specifically to T1.
    #par =list([td,num_gen,std_g1,std_bud,f,k,r,T1])
    delta=par[5] * par[0] * np.log(1 + par[6])/ np.log(2)
    for i in range(par[1]):
        if not(par[2]==0):
            noise_g1=np.random.normal(0,par[2],d.shape[1]+m.shape[1]) # Noise for both mother and daughter populations
        else: noise_g1=np.zeros(d.shape[1]+m.shape[1])
        if not(par[3] == 0):
            noise_bud = np.random.normal(0, par[3], d.shape[1] + m.shape[1]) # Noise for both mother and daughter populations
        else: noise_bud=np.zeros(d.shape[1]+m.shape[1])
            # update the abundance of Whi5 for the next generation, assuming noise in passage through Start
        d_temp = np.empty([d.shape[0],m.shape[1]+d.shape[1]]) # temporary array to store the whi5 and volume measurements.
        m_temp = np.empty([m.shape[0],
                       m.shape[1] + d.shape[1]])  # temporary array to store the whi5 and volume measurements.
        # store the Whi5 data for daughters born from mother cells first
        d_temp[1,:m.shape[1]] = par[4] * (m[1, :] + delta + noise_bud[:m.shape[1]] * par[5])
        # Now store the values for daughters born from daughters
        d_temp[1,m.shape[1]:] =par[4] * (d[1, :] + delta + noise_bud[m.shape[1]:d.shape[1] + m.shape[1]] * par[5])
        # Mother amount of Whi5 related to daughter amount by (1-f)/f. The fact that we have noise in budded phase
        # shouldn't affect this
        m_temp[1,:] = (1 - par[4]) / par[4] * d_temp[1,:]
        # store the volumes for the mothers born from mothers first. Update directly from the amount of Whi5 the cell has
        # since Whi5 is measured in units of volume here. Have to do from previous generation because we look at Vb
        # amount of Whi5 is deterministic in setting timing of G1, and then add noise on top.
        # Note that this assumes the dilution model for passage through Start, and hence the incremental model for mother growth.
        m_temp[0,:m.shape[1]] = np.max(m, 0) * np.exp((par[7]+noise_g1[:m.shape[1]])* np.log(2) / par[0])
        # same for mothers born from daughters. Assumes incremental model for daughter growth.
        m_temp[0,m.shape[1]:] = np.max(d, 0) * np.exp((par[7]+
            noise_g1[m.shape[1]:d.shape[1] + m.shape[1]]) * np.log(2) / par[0])
        # We use mother volume to also define the volume of the next gen. daughter. Vs(2^(tn/Td)*(1+r)-1)
        d_temp[0,:] = m_temp[0,:] * (np.exp(noise_bud[:] * np.log(2) / par[0]) * (1 + par[6]) - 1)
        # Ordering is consistent with that for Whi5, with daughters born from mother cells first, and daughters born from daughters second.
        del m,d
        m=np.empty(m_temp.shape)
        d=np.empty(d_temp.shape)
        m[:,:]=m_temp[:,:]
        d[:,:]=d_temp[:,:]
        del m_temp,d_temp
    return d,m

def timr_mothers_simulator(d,m,par): # this differs from the above incr_mothers_simulator in that it includes a period T1
    # after initiating Start but before budding occurs. We assume that T1 is the same for daughters and mothers.
    # This is the same as for incr_mothers_simulator except that we shorten the period prior to the budded phase. The noise
    # in g1 is added specifically to T1.
    #par =list([td,num_gen,std_g1,std_bud,f,delta,k,r,T1])
    delta=par[5] * par[0] * np.log(1 + par[6])/ np.log(2)
    for i in range(par[1]):
        if not(par[2]==0):
            noise_g1=np.random.normal(0,par[2],d.shape[1]+m.shape[1]) # Noise for both mother and daughter populations
        else: noise_g1=np.zeros(d.shape[1]+m.shape[1])
        if not(par[3] == 0):
            noise_bud = np.random.normal(0, par[3], d.shape[1] + m.shape[1]) # Noise for both mother and daughter populations
        else: noise_bud=np.zeros(d.shape[1]+m.shape[1])
            # update the abundance of Whi5 for the next generation, assuming noise in passage through Start
        d_temp = np.empty([d.shape[0],m.shape[1]+d.shape[1]]) # temporary array to store the whi5 and volume measurements.
        m_temp = np.empty([m.shape[0],
                       m.shape[1] + d.shape[1]])  # temporary array to store the whi5 and volume measurements.
        # store the Whi5 data for daughters born from mother cells first
        d_temp[1,:m.shape[1]] = par[4] * (m[1, :] + delta + noise_bud[:m.shape[1]] * par[5])
        # Now store the values for daughters born from daughters
        d_temp[1,m.shape[1]:] =par[4] * (d[1, :] + delta + noise_bud[m.shape[1]:d.shape[1] + m.shape[1]] * par[5])
        # Mother amount of Whi5 related to daughter amount by (1-f)/f. The fact that we have noise in budded phase
        # shouldn't affect this
        m_temp[1,:] = (1 - par[4]) / par[4] * d_temp[1,:]
        # store the volumes for the mothers born from mothers first. Update directly from the amount of Whi5 the cell has
        # since Whi5 is measured in units of volume here. Have to do from previous generation because we look at Vb
        # amount of Whi5 is deterministic in setting timing of G1, and then add noise on top.
        m_temp[0,:m.shape[1]] = m[0,:] * np.exp((par[7]+noise_g1[:m.shape[1]])* np.log(2) / par[0]) # mother cells don't delay Start due to Whi5
        m_temp[0,m.shape[1]:] = np.max(d, 0) * np.exp((par[7]+
            noise_g1[m.shape[1]:d.shape[1] + m.shape[1]]) * np.log(2) / par[0]) # Assumes incremental model for daughter growth.
        # We use mother volume to also define the volume of the next gen. daughter. Vs(2^(tn/Td)*(1+r)-1)
        d_temp[0,:] = m_temp[0,:] * (np.exp(noise_bud[:] * np.log(2) / par[0]) * (1 + par[6]) - 1)
        # Ordering is consistent with that for Whi5, with daughters born from mother cells first, and daughters born from daughters second.
        del m,d
        m=np.empty(m_temp.shape)
        d=np.empty(d_temp.shape)
        m[:,:]=m_temp[:,:]
        d[:,:]=d_temp[:,:]
        del m_temp,d_temp
    return d,m

def incr_mothers_simulator2(d,m,par):
    # Same as incr_mothers_simulator1 except that it accounts for the possible destruction of Whi5
    # immediately after passage through Start, with a factor g which is the fraction of Whi5 retained. Could probably speed up by using
    # vdfrac_temp defined based on volumes at birth.
    # par =list([td,num_gen,std_g1,std_bud,f,k,r,T1,g])
    delta=par[5] * par[0] * np.log(1 + par[6])/ np.log(2)
    for i in range(par[1]):
        if not(par[2]==0):
            noise_g1=np.random.normal(0,par[2],d.shape[1]+m.shape[1]) # Noise for both mother and daughter populations
        else: noise_g1=np.zeros(d.shape[1]+m.shape[1])
        if not(par[3] == 0):
            noise_bud = np.random.normal(0, par[3], d.shape[1] + m.shape[1]) # Noise for both mother and daughter populations
        else: noise_bud=np.zeros(d.shape[1]+m.shape[1])
            # update the abundance of Whi5 for the next generation, assuming noise in passage through Start
        d_temp = np.empty([d.shape[0],m.shape[1]+d.shape[1]]) # temporary array to store the whi5 and volume measurements.
        m_temp = np.empty([m.shape[0],
                       m.shape[1] + d.shape[1]])  # temporary array to store the whi5 and volume measurements.
        # store the Whi5 data for daughters born from mother cells first
        d_temp[1,:m.shape[1]] = par[4] * (par[8]*m[1, :] + delta + noise_bud[:m.shape[1]] * par[5])
        # Now store the values for daughters born from daughters
        d_temp[1,m.shape[1]:] =par[4] * (par[8]*d[1, :] + delta + noise_bud[m.shape[1]:d.shape[1] + m.shape[1]] * par[5])
        # Mother amount of Whi5 related to daughter amount by (1-f)/f. The fact that we have noise in budded phase
        # shouldn't affect this
        m_temp[1,:] = (1 - par[4]) / par[4] * d_temp[1,:]
        # store the volumes for the mothers born from mothers first. Update directly from the amount of Whi5 the cell has
        # since Whi5 is measured in units of volume here. Have to do from previous generation because we look at Vb
        # amount of Whi5 is deterministic in setting timing of G1, and then add noise on top.
        # Note that this assumes the dilution model for passage through Start, and hence the incremental model for mother growth.
        m_temp[0,:m.shape[1]] = np.max(m, 0) * np.exp((par[7]+noise_g1[:m.shape[1]])* np.log(2) / par[0])
        # same for mothers born from daughters. Assumes incremental model for daughter growth.
        m_temp[0,m.shape[1]:] = np.max(d, 0) * np.exp((par[7]+
            noise_g1[m.shape[1]:d.shape[1] + m.shape[1]]) * np.log(2) / par[0])
        # We use mother volume to also define the volume of the next gen. daughter. Vs(2^(tn/Td)*(1+r)-1)
        d_temp[0,:] = m_temp[0,:] * (np.exp(noise_bud[:] * np.log(2) / par[0]) * (1 + par[6]) - 1)
        # Ordering is consistent with that for Whi5, with daughters born from mother cells first, and daughters born from daughters second.
        del m,d
        m=np.empty(m_temp.shape)
        d=np.empty(d_temp.shape)
        m[:,:]=m_temp[:,:]
        d[:,:]=d_temp[:,:]
        del m_temp,d_temp
    return d,m


def incr_mothers_simulator3(d,m,par): # Same as incr_mothers_simulator2 except that it makes the amount of Whi5 received
    # proportional to the volume fraction ACCOUNTING FOR NOISE IN THE G2 PERIOD.
    # par =list([td,num_gen,std_g1,std_bud,f,k,r,T1,g]) # Note f is just a placeholder here!
    delta=par[5] * par[0] * np.log(1 + par[6])/ np.log(2)
    for i in range(par[1]):
        if not(par[2]==0):
            noise_g1=np.random.normal(0,par[2],d.shape[1]+m.shape[1]) # Noise for both mother and daughter populations
        else: noise_g1=np.zeros(d.shape[1]+m.shape[1])
        if not(par[3] == 0):
            noise_bud = np.random.normal(0, par[3], d.shape[1] + m.shape[1]) # Noise for both mother and daughter populations
        else: noise_bud=np.zeros(d.shape[1]+m.shape[1])
            # update the abundance of Whi5 for the next generation, assuming noise in passage through Start
        d_temp = np.empty([d.shape[0],m.shape[1]+d.shape[1]]) # temporary array to store the whi5 and volume measurements.
        m_temp = np.empty([m.shape[0],
                       m.shape[1] + d.shape[1]])  # temporary array to store the whi5 and volume measurements.
        # Here we compute volumes first
        # store the volumes for the mothers born from mothers first. Update directly from the amount of Whi5 the cell has
        # since Whi5 is measured in units of volume here. Have to do from previous generation because we look at Vb
        # amount of Whi5 is deterministic in setting timing of G1, and then add noise on top.
        # Note that this assumes the dilution model for passage through Start, and hence the incremental model for mother growth.
        m_temp[0, :m.shape[1]] = np.max(m, 0) * np.exp((par[7] + noise_g1[:m.shape[1]]) * np.log(2) / par[0])
        # same for mothers born from daughters. Assumes incremental model for daughter growth.
        m_temp[0, m.shape[1]:] = np.max(d, 0) * np.exp((par[7] +
                                                        noise_g1[m.shape[1]:d.shape[1] + m.shape[1]]) * np.log(2) / par[
                                                           0])
        # We use mother volume to also define the volume of the next gen. daughter. Vs(2^(tn/Td)*(1+r)-1)
        d_temp[0, :] = m_temp[0, :] * (np.exp(noise_bud[:] * np.log(2) / par[0]) * (1 + par[6]) - 1)
        # Ordering is consistent with that for Whi5, with daughters born from mother cells first, and daughters born from daughters second.

        # Now we calculate the volume fractions of the daughter relative to mother + daughter
        vdfrac_temp=np.empty(m.shape[1]+d.shape[1])
        vdfrac_temp[:]=d_temp[0, :]/(d_temp[0, :]+m_temp[0, :])
        # store the Whi5 data for daughters born from mother cells first
        d_temp[1,:m.shape[1]] = vdfrac_temp[:m.shape[1]]* (par[8]*m[1, :] + delta + noise_bud[:m.shape[1]] * par[5])
        # Now store the values for daughters born from daughters
        d_temp[1,m.shape[1]:] =vdfrac_temp[m.shape[1]:] * (par[8]*d[1, :] + delta + noise_bud[m.shape[1]:d.shape[1] + m.shape[1]] * par[5])
        # Mother amount of Whi5 related to daughter amount by (1-f)/f. The fact that we have noise in budded phase
        # shouldn't affect this
        m_temp[1,:] = (1 - vdfrac_temp[:])/vdfrac_temp[:] * d_temp[1,:]
        del m,d,vdfrac_temp
        m=np.empty(m_temp.shape)
        d=np.empty(d_temp.shape)
        m[:,:]=m_temp[:,:]
        d[:,:]=d_temp[:,:]
        del m_temp,d_temp
    return d,m



    # time greater than the time of the simulation will not divide). In doing so we calculate the time to division etc.
########################################################################################################################
# DISCRETIZED TIME SIMULATIONS
########################################################################################################################


# Now let's deal with this issue of discretized time simulations. First we define our classes for mother and daughter
# cells.

# This defines a global variable within growth_simulation which is necessary to define the Mother and Daughter classes.
# Anything that is defined in this variable is fixed for the course of the simulation. If you wish to iterate over a v
# variable you should include that as an input parameter for the function discr_time1
par = dict([('td', 1.0), ('num_s', 100), ('Vo', 1.0), ('std_iv', 0.1),
            ('std_iw', 0.1), ('delta_v', 1.0), ('k', 1.0)])
# dt=timestep
# nstep=number of timesteps we go through
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
    K = par['k']   # production rate of Whi5 (produces one unit of Whi5 in time taken for cell mass to double).
    cellCount = 0  # total number of cells
    delta_v = par['delta_v']

    def __init__(self, vb, whi5, tb, mother=None, dilution=True):
        # produces a new instance of the mother class. Only instance variables
        # initially are the whi5 abundance at birth, the volume at birth, the time of birth and it points to the mother.

        self.vb = vb
        self.wb = whi5
        self.tb = tb
        self.mother = mother  # references the mother cell
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.isdaughter = False  # boolean for whether the cell is a mother or daughter
        self.dilution = dilution  # stores the kind of growth policy assumed for this cell
        self.exists = True  # indexes whether the cell exists currently
        Cell.cellCount += 1

    def grow(self, par1):  # dilution model for whi5 for both mother and daughter cells.
        self.R = par1['r']  # asymmetry ratio
        self.t_delay = par1['t_delay']
        # self.delta = self.K * np.log(1 + self.R) / np.log(2)  # delta in whi5 during growth phase
        if self.dilution:
            # Calculate the volume of this cell at start, volume at division, and whi5 at division.
            if par1['g1_thresh_std'] != 0:
                self.noise_thresh = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
            else:
                self.noise_thresh = 0.0
            if par1['g1_std'] != 0:
                self.noiseg1 = np.random.normal(0.0, par1['g1_std'], 1)[0]
                # generate and retain the noise in G2 for each instance of the mother class
            else:
                self.noiseg1 = 0.0
            # note that noise is measured as a fraction of growth rate.
            # self.vs = np.maximum(np.maximum(self.vb, self.wb) * 2 ** (self.noiseg1 + self.t_delay),self.vb)
            self.vs = (self.wb/(1+self.noise_thresh)) * np.exp(self.noiseg1*np.log(2)/self.td) * 2 ** self.t_delay

            if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                self.noiseg2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
            else:
                self.noiseg2 = 0.0
            self.vd = self.vs * (1.0 + self.R) * np.exp(self.noiseg2*np.log(2)/self.td)
            # self.wd = self.wb + np.maximum(np.log(self.vd / self.vs) * self.K / np.log(2), 0.0)
            self.wd = self.wb + self.K * (np.log(1.0 + self.R) * self.td / np.log(2.0) + self.noiseg2)
            self.t_grow = np.maximum(np.log(self.vd / self.vb) * self.td / np.log(2), 0.0)
            self.t_div = self.tb + self.t_grow
        else:
            # This function takes a step back, simply considering
            # addition of a constant volume delta from birth to division.
            # No consideration is given to passage through Start
            # Noise in g1 = size additive. Noise in g2 = time additive.
            # std dev in additive noise in division time
            # standard deviation in multiplicative noise in division time.
            if par1['g1_std'] != 0:  # additive noise in size.
                self.noiseg1 = np.random.normal(0.0, par1['g1_std'], 1)[
                    0]  # generate and retain the noise in G2 for each instance of the mother class
            else:
                self.noiseg1 = 0.0
            if par1['g2_std'] != 0.0:  # multiplicative noise in division time.
                self.noiseg2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
            else:
                self.noiseg2 = 0.0
            self.vs = (self.vb + self.delta_v) * np.exp(self.noiseg1*np.log(2)/self.td) / (1+self.R)
            self.vd = self.vs * (1+self.R) * np.exp(self.noiseg2*np.log(2)/self.td)
            self.t_grow = self.td * np.log(self.vd/self.vb)/np.log(2)
            self.t_div = self.tb + self.t_grow

    # here par is a dictionary containing all the relevant parameters, a global variable defined in growth_simulation.
def starting_popn(par1, dilution=True):  # here the boolean optional parameter 'dilution' specifies whether the model involved
    # should be based on a Whi5 dilution scheme.
    # par=dict([('dt', 1),('nstep',100), ('td', 90), ('num_s', 100),('Vo',1.0),('std_iv',0.1),('std_iw',0.1)])
    # Initialise simulation with a normal distribution of cell sizes and Whi5 abundances.
    v_init = np.random.normal(loc=par['Vo'], scale=par['std_iv'], size=par['num_s'])
    # we will try to initialize the concentrations of whi5 such that it is consistent with a dilution model on average.
    tb = np.log(1 + par1['r']) * par['td'] / np.log(2)
    w_init = par1['w_f'] * (v_init/par1['r'] + par['k'] * tb)
    # W_init = np.random.normal(loc=par['Vo'], scale=par['std_iw'], size=par['num_s'])
    del tb
    t_init = np.random.uniform(low=0.0, high=1.0, size=par['num_s'])
    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []
    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        # Half are treated as daughters for the purposes of this simulation (since every odd cell has the previous even
        # cell as a daughter).
        c.append(Cell(v_init[i], w_init[i], 0, dilution=dilution))
        if np.mod(i, 2) == 0:
            c[-1].isdaughter = True  # every second cell will be a daughter cell.
        c[-1].grow(par1)
        c[-1].t_div = t_init[i]*c[-1].t_div
        c[-1].tb = c[-1].t_div-c[-1].t_grow
        # we expect that these cells have been caught at
        # some random point of progression through their cell cycles.
    del v_init, w_init
    return c

def next_gen(index, f, time, par1):
    if f[index].dilution:
        # This function resets growth-policy specific variables for a single birth event in a whi5 dilution manner.
        # Should be used within discr_time to evolve the list of cells c.
        if par1['w_f'] == par1['r']/(par1['r']+1):
            wd = f[index].wd*(f[index].vd-f[index].vs)/f[index].vd  # volumetric fraction of whi5 given
        else:
            wd = par1['w_f'] * f[index].wd
            raise ValueError('ya done fucked up')
        f.append(Cell(f[index].vd - f[index].vs, wd, time, mother=weakref.proxy(f[index])))
        # Produce a new cell based on the previous one and append it to the end of the list.
        f[-1].grow(par1)  # grow newborn cell
        f[-1].isdaughter = True  # track that this cell is a daughter
        f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
        # add new cell for newborn mother cell.
        f.append(Cell(f[index].vs, f[index].wd-wd, time, mother=weakref.proxy(f[index])))
        f[-1].grow(par1)  # grow newborn cell
        f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
        f[index].exists = False  # track that this cell no longer "exists".
    else:
        # This function resets growth-policy specific variables for a single birth event in a whi5 independent manner.
        # Should be used within discr_time to evolve the list of cells c.
        f.append(Cell(f[index].vd - f[index].vs, 0.0, time, mother=weakref.proxy(f[index]), dilution=False))
        # Produce a new cell based on the previous one and append it to the end of the list.
        f[-1].grow(par1)  # grow newborn cell
        f[-1].isdaughter = True  # track that this cell is a daughter
        f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
        # add new cell for newborn mother cell.
        f.append(Cell(f[index].vs, 0.0, time, mother=weakref.proxy(f[index]), dilution=False))
        f[-1].grow(par1)  # grow newborn cell
        f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
        f[index].exists = False  # track that this cell no longer "exists"
    return f

def discr_time(par1, dilution):
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
    c = starting_popn(par1, dilution=dilution)
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
        #if np.mod(i,100) == 0:
        #        print('Time step: ', i)
    return c, num_cells, tvec


def simulation_stats(d,m):
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

def dplot(x,y,st):
    vals=scipy.stats.linregress(x,y)
    xmax = np.mean(x) + 2.5 * np.std(x)
    xmin = np.mean(x) - 2.5 * np.std(x)
    ymin = np.mean(y) - 2.5 * np.std(y)
    ymax = np.mean(y) + 2.5 * np.std(y)
    xv = np.array([xmin, xmax])
    plt.figure(figsize=[10, 10])
    plt.hexbin(x, y, cmap="Purples",gridsize=50)
    plt.plot(xv, vals[0] * xv + vals[1], label='slope= ' + '%.3f' % (vals[0]) + '$\pm$''%.3f' % (vals[4]))
    plt.title(st, size=30, y=1.04)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.legend(fontsize = 'x-large')
    plt.xlabel('$V_b$',size=30,weight='bold')
    plt.ylabel('$V_d$',size=30,weight='bold')
    plt.show()
    return vals

def single_cycle(par1, num_cells, dilution=True):
    V_init = 1.0  # starting volume for population of num_cells
    W_init=np.random.normal(loc=1.0, scale=0.2, size=num_cells)
    #W_init = np.ones(num_cells)  # starting Whi5 amount for population of num_cells
    f = []
    for i in range(num_cells):  # instantiating our initial population of cells. These cells do not have mothers.
        # Half are treated as daughters for the purposes of this simulation (since every odd cell has the previous even
        # cell as a daughter).
        f.append(Cell(V_init, W_init[i], 0, dilution=dilution))
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
            # Should be used within discr_time to evolve the list of cells c.
            c.append(Cell(f[index].vd - f[index].vs, 0.0, 0.0,dilution=False))
            # Produce a new cell based on the previous one and append it to the end of the list.
            c[-1].grow(par1)  # grow newborn cell
            c[-1].isdaughter = True  # track that this cell is a daughter
            # add new cell for newborn mother cell.
            c.append(Cell(f[index].vs, 0.0, 0.0, dilution=False))
            c[-1].grow(par1)  # grow newborn cell
    del V_init, W_init
    return c

