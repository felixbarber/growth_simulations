#!/usr/bin/env python

import os
import numpy as np
import time
from scipy import stats
import scipy

rank = os.environ["PARAM1"]
print rank
temp = np.random.normal(0, 1.0, 100)
np.save('temp_{0}'.format(rank))