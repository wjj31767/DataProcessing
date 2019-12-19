from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('cc3d.csv')
fig = plt.figure()
datasize = data["delimeter"]*(10**4)
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data["meanx"],
               data["meany"],
               data["meanz"],
               c=datasize,
               marker='o',
           s=datasize)
plt.show()