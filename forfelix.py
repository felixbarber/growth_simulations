#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


def function(x,y):
	return x*y


if __name__=='__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	print "I am processor {0} and I am hungry!".format(rank)
	if rank == 0:
		print 'Notice that they are out of order, and if you run it over and over the order changes.'
	#KISS first. Make everyone make a matrix
	X , Y = 1000 , 1000
	a = np.zeros((X,Y,3,4))
	if rank == 0:
		print 'Setup matrix'
	dx = X / size
	start = dx * rank
	stop = start + dx
	if rank == size-1:
		stop = X
	for i in xrange(start,stop):
		for j in range(Y):
			a[i][j] = function(i,j)
	##Now everyone has made part of the matrix. Send to one processor. Many ways to do this. Broadcast, sendrecv etc
	if rank != 0:
		comm.send(a, dest=0)
	if rank == 0:
		new_grid = np.zeros(np.shape(a))
		new_grid += a
		for p in range(1,size):
			print 'I am 0 and I got from ',p
			new_grid += comm.recv(source = p)
	comm.barrier()
	if rank == 0:
		print 'Done!'
		#plt.imshow(new_grid)
		#plt.show()
		np.save('test1',new_grid)
