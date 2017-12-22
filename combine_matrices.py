#!/usr/bin/env python

import os
import numpy as np

path = os.environ["FILE_BASE"]
num = os.environ["NUM_FILES"]

for i0 in range(int(num)):
    temp = np.load(path+'{0}.npy'.format(i0))
    if i0 == 0:
        a = np.zeros(temp.shape)
    a += temp
    del temp

np.save(path+'complete.npy', a)
