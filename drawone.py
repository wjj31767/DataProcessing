import numpy as np
from scipy import interpolate
import pandas as pd
import os
import time
import multiprocessing
import matplotlib.pyplot as plt
nn = pd.read_csv('a05txt',header=None,skiprows=[0,1,2],sep='\s+',)
plt.scatter(nn[0] / 100000, nn[1] / 100000, c=nn[2])
plt.colorbar()
plt.xlim((nn[0].min() - 1) / 100000, (nn[0].max() + 1) / 100000)
plt.ylim((nn[1].min() - 10) / 100000, (nn[1].max() + 10) / 100000)
plt.show()