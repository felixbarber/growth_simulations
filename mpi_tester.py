#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import time
from scipy import stats
import scipy

# This script will produce a random variable then save that as a numpy array

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_rep = 4 * size + 3
    file = open('./mpi_tester/out_{0}'.format(rank), 'w')
    file.write('Hello world, this is processor number {0}'.format(rank))
    file.close()

    X = [num_rep, 2]
    a = np.empty(X)

    if rank == 0:
        tic = time.clock()

    dx = num_rep/ size
    rem = np.mod(num_rep, size)
    if rank >= size - rem:  # this makes sure that it distributes the remainder as equally as possible.
        start = dx * (size - rem) + (dx + 1) * (rank + rem - size)
        stop = start + dx + 1
    else:
        start = dx * rank
        stop = start + dx
    if rank == size - 1:
        stop = num_rep

    for i0 in xrange(start, stop):
        temp = np.random.normal(loc=0.0, scale=1.0, size=20)+np.linspace(0.0, 10.0, 20)
        temp1 = np.linspace(0.0, 10.0, 20)
        temp2 = scipy.stats.linregress(temp1, temp)
        a[i0, 0] = temp2[0]
        a[i0, 1] = temp2[1]

    if rank != 0:
        comm.send(a, dest=0)
    if rank == 0:
        new_grid = np.zeros(np.shape(a))
        new_grid += a
        for p in range(1, size):
            print 'I am 0 and I got from ', p
            new_grid += comm.recv(source=p)
    comm.barrier()
    if rank == 0:
        print 'Time taken =', time.clock()-tic
        # plt.imshow(new_grid)
        # plt.show()
        np.save('./mpi_tester/output_file', new_grid)
