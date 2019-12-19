import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
drawforhist = np.genfromtxt('cc3d.csv',delimiter=',',skip_header=1)
drawforhist = drawforhist[:,0]
print(drawforhist.min(),drawforhist.max())
binnum = 100
bin_edges = np.arange( 0.00005 ,0.0001,0.00005/binnum)
plt.hist(drawforhist,bins=bin_edges,color='red')
plt.show()