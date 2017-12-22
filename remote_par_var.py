import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g

par1 = dict([('g1_std', 0.0), ('g2_std', 0.05),('g1_thresh_std',0.05),('nstep', 1200),('dt', 0.01),('t_delay',0.0)\
             ,('initiator', 0),('CD',0.75)])
obs = g.parameter_variation_asymmetric(par1)
np.save('par_var_output', obs)
