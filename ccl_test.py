import numpy as np
from scipy.interpolate import griddata
import pandas
import skimage
import os
import sys
import matplotlib.pyplot as plt

t = np.genfromtxt('boxTest.csv',delimiter=',',skip_header=1)
print(len(t))
xi = t[:,6].min()
xx = t[:,6].max()
yi = t[:,7].min()
yx = t[:,7].max()
zi = t[:,5].min()
zx = t[:,5].max()
PointZSet = set()
for i in t[:,5]:
    PointZSet.add(i)
path = "zz"
for PointZ in PointZSet:
    filename = os.path.join(path,str(PointZ))
    print(t[t[:,5]==PointZ])
    # TODO: mb use pandas doint these things better. and mb I don't need set() here.
# grid_x, grid_y, grid_rz= np.mgrid[xi:xx:1e-5, yi:yx:1e-5, zi:zx:1e-5]
# points = t[:,[6,7,5]]
# values = t[:,1]
# alpha = t[:,0]
# grid_z = griddata(points, values, (grid_x, grid_y,grid_rz), method='nearest')
# grid_alpha = griddata(points, alpha, (grid_x, grid_y,grid_rz), method='nearest')
# print(grid_z)
# grid_z[grid_z<0.5]=0
# grid_z[grid_z>=0.5] = 1
# grid_z = grid_z.astype(np.int64)
#
# sh = skimage.measure.label(grid_z,connectivity = 1, return_num=True)
# newfile = open('ccl.csv','w')
# for i in range(1,sh[1]):
#     x,y=np.where(sh[0]==i)
#     tsum = 0
#     for j in zip(x,y):
#         tsum+=grid_alpha[j[0],j[1]]
#     newfile.write(str(i)+','+str(tsum)+'\n')
# newfile.close()

import pandas as pd
# TODO: mb use pandas doing these things better. and mb I don't need set() here.
# TODO: 1. dividing all the files by Z

boxTextReadFromPandas = pd.read_csv('boxTest.csv')
path = 'ccl_clip'
getClipFileNum = 0
for partdividedbyPointZ in boxTextReadFromPandas.groupby("Points:0"):
    # if len(partdividedbyPointZ)>1:
    #     # print(partdividedbyPointZ)
    #     # print(partdividedbyPointZ[0])

    if getClipFileNum==2:
        break
    path4Clip = os.path.join(path,str(partdividedbyPointZ[0])+'.csv')
    print(path4Clip)
    partdividedbyPointZ[1].to_csv(path4Clip)
    getClipFileNum += 1
# TODO: 2. calculating the 2d ccl analysis
for t in os.pathdir(path):
    t = np.genfromtxt('boxTest.csv',delimiter=',',skip_header=1)
    xi = t[:,6].min()
    xx = t[:,6].max()
    yi = t[:,7].min()
    yx = t[:,7].max()
    # grid_x, grid_y, grid_rz= np.mgrid[xi:xx:1e-5, yi:yx:1e-5, zi:zx:1e-5]
    # points = t[:,[6,7,5]]
    # values = t[:,1]
    # alpha = t[:,0]
    # grid_z = griddata(points, values, (grid_x, grid_y,grid_rz), method='nearest')
    # grid_alpha = griddata(points, alpha, (grid_x, grid_y,grid_rz), method='nearest')
    # print(grid_z)
    # grid_z[grid_z<0.5]=0
    # grid_z[grid_z>=0.5] = 1
    # grid_z = grid_z.astype(np.int64)
    #
    # sh = skimage.measure.label(grid_z,connectivity = 1, return_num=True)
    # newfile = open('ccl.csv','w')
    # for i in range(1,sh[1]):
    #     x,y=np.where(sh[0]==i)
    #     tsum = 0
    #     for j in zip(x,y):
    #         tsum+=grid_alpha[j[0],j[1]]
    #     newfile.write(str(i)+','+str(tsum)+'\n')
    # newfile.close()


# TODO: 2.1 can the mesh start and end part just the same ?
# TODO: 3. merge them all



# prop = skimage.measure.regionprops(sh[0])
