
import numpy as np
import matplotlib.pyplot as plt
drawforhist = np.genfromtxt('cc3d.csv',delimiter=',',skip_header=1)
drawforhist = drawforhist[:,0]
mi = drawforhist.min()
mx = drawforhist.max()
print(mi,mx)
binnum = 100
bin_edges = np.arange( mi ,mx,(mx-mi)/binnum)
plt.hist(drawforhist,bins=bin_edges,color='red')
plt.show()