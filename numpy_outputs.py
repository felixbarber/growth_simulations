#!/usr/bin/env python

import os
import numpy as np

path = os.environ["FILE_BASE"]
num = os.environ["NUM_FILES"]

a = np.zeros([int(num)])

for i0 in range(int(num)):
    temp = np.load(path+'{0}.npy'.format(i0))
    print num, temp[0]
    a[i0] = temp[0]
    del temp

np.save(path+'complete.npy', a)
