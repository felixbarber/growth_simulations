#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def function(x, sx, y, sy):
    return np.exp(0.5*((x*sy)**2+(y*sx)**2+2*x*y)/(1-(sx*sy)**2))/np.sqrt(1-sx**2*sy**2)

sx = np.linspace(0.01, 0.99, 99)
sy = np.linspace(0.01, 0.99, 5)

ux = np.linspace(-0.4, 0.4, 2)
uy = np.linspace(-0.4, 0.4, 2)

num = 10**5

fig = plt.figure(figsize=[20, 20])
for i in range(len(uy)):
    for j in range(len(ux)):
        ax = fig.add_subplot(len(uy), len(ux), i*len(ux)+j+1)
        for k in range(len(sy)):
            vec = np.zeros(len(sx))
            vec1 = np.zeros(len(sx))
            valsy = np.random.normal(uy[i], sy[k], num)
            for h in range(len(sx)):
                valsx = np.random.normal(ux[j], sx[h], num)
                prod = np.asarray([valsx[l]*valsy[l] for l in range(num)])
                del l
                vec[h] = np.mean(np.exp(prod))
                vec1[h] = function(ux[j], sx[h], uy[i], sy[k])
                del prod
            ax.plot(sx, vec, label='Sim $\sigma_{y}=$'+str(np.round(sy[k], 2)))

            ax.plot(sx, vec1, label='Theory $\sigma_{y}=$'+str(np.round(sy[k], 2)))
            ax.set_yscale('log')
            del vec, vec1
        ax.legend(loc=2)
        ax.set_title('$\mu_{x}=$' + str(np.round(ux[j], 2)) + ' $\mu_{y}=$' +
                     str(np.round(uy[i], 2)))
        # if i == len(uy) - 1:
        #     ax.set_xlabel('$<e^{xy}>$')
        if j == 0:
            ax.set_ylabel('$\sigma_x$')
        print ('finished one plot')
# plt.suptitle('Test of expression for $<e^{xy}>$')
fig.savefig('./function_tester.eps', bbox_inches='tight', dpi=fig.dpi)
