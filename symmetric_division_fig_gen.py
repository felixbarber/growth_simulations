#!/usr/bin/env python

import growth_simulation_yeast_1 as g

par1 = dict([('g1_std', 0.0), ('g2_std', 0.), ('g1_thresh_std', 0.05), ('nstep', 1200), ('dt', 0.01), ('t_delay', 0.0) \
                , ('initiator', 0), ('CD', 0.75), ('num_gen', 9), ('K', 1.0), ('td', 1.0)])

slope,fig = g.single_par_test(par1)
fig.savefig('tree_discr_time_symmdiv_noiselessadder_vdvb.eps', dpi=fig.dpi)


