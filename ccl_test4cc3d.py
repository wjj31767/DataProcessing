import cc3d
import numpy as np
import cupy as cp
import pandas as pd
from scipy.interpolate import griddata
import datetime
import math

labels_in = []
Vmatrix = []
alphaMatrix = []
Xmatrix = []
Ymatrix = []
Zmatrix = []
print(datetime.datetime.now())

boxTextReadFromPandas = pd.read_csv('boxTest.csv')

xi = boxTextReadFromPandas['Points:1'].min()
xx = boxTextReadFromPandas['Points:1'].max()
yi = boxTextReadFromPandas['Points:2'].min()
yx = boxTextReadFromPandas['Points:2'].max()

print(xi, xx, yi, yx)

Thread_value = 0.005
gridvalue = 25e-6

with open('cc3d.csv', 'w') as newfile:
    newfile.write("delimeter,Volumn,meanx,meany,meanz\n")

    n = 0

    for num, partdividedbyPointZ in enumerate(boxTextReadFromPandas.groupby("Points:0")):
        # print(num,partdividedbyPointZ[0])
        partdividedbyPointZ = partdividedbyPointZ[1].to_numpy()

        points = partdividedbyPointZ[:, [6, 7]]
        alpah_liquid = partdividedbyPointZ[:, 1]
        V = partdividedbyPointZ[:, 0]
        X = partdividedbyPointZ[:, 6]
        Y = partdividedbyPointZ[:, 7]
        Z = partdividedbyPointZ[:, 5]
        grid_x, grid_y = np.mgrid[xi:xx:gridvalue, yi:yx:gridvalue]
        grid_z = griddata(points, alpah_liquid, (grid_x, grid_y), method='nearest')
        grid_alpha = griddata(points, alpah_liquid, (grid_x, grid_y), method='nearest')
        grid_V = griddata(points, V, (grid_x, grid_y), method='nearest')
        grid_corx = griddata(points, X, (grid_x, grid_y), method='nearest')
        grid_cory = griddata(points, Y, (grid_x, grid_y), method='nearest')
        grid_corz = griddata(points, Z, (grid_x, grid_y), method='nearest')
        grid_z[grid_z < Thread_value] = 0
        grid_z[grid_z >= Thread_value] = 1
        grid_z = grid_z.astype(np.int32)

        if num % 1000 == 0:
            print(datetime.datetime.now(), (num + 1) / 14297)
        grid_z = cp.asarray(grid_z)
        if cp.sum(grid_z) == 0:
            print(n)
            n += 1

            labels_in = np.asarray(labels_in)

            if np.sum(labels_in) == 0:
                labels_in = []
                Vmatrix = []
                alphaMatrix = []
                Xmatrix = []
                Ymatrix = []
                Zmatrix = []
                continue

            # Vmatrix = np.power(np.asarray(Vmatrix),1/3)
            labels_out = cc3d.connected_components(labels_in)
            N = np.max(labels_out)
            for segid in range(1, N + 1):
                # calculate delimeter
                Vmatrix = cp.asarray(Vmatrix)
                alphaMatrix = cp.asarray(alphaMatrix)
                tmpMatrix = cp.asarray(labels_out == segid)
                sumtmpMatrix = cp.multiply(Vmatrix, cp.multiply(alphaMatrix, tmpMatrix))
                sumVolumn = cp.sum(sumtmpMatrix)  # sum of Volumn
                sumVolumnDelimeter = math.pow(sumVolumn * 6 / np.pi, 1 / 3)

                # coordinate
                Xmatrix = cp.asarray(Xmatrix)
                Ymatrix = cp.asarray(Ymatrix)
                Zmatrix = cp.asarray(Zmatrix)
                meanx = cp.sum(cp.multiply(Xmatrix, sumtmpMatrix)) / sumVolumn
                meany = cp.sum(cp.multiply(Ymatrix, sumtmpMatrix)) / sumVolumn
                meanz = cp.sum(cp.multiply(Zmatrix, sumtmpMatrix)) / sumVolumn
                # write into file
                newfile.write(str(sumVolumnDelimeter)
                              + ',' + str(sumVolumn)
                              + ',' + str(meanx)
                              + ',' + str(meany)
                              + ',' + str(meanz)
                              + '\n')

            labels_out = []
            labels_in = []
            alphaMatrix = []
            Vmatrix = []
            Xmatrix = []
            Ymatrix = []
            Zmatrix = []
        else:
            labels_in.append(grid_z.tolist())
            alphaMatrix.append(grid_alpha.tolist())
            Vmatrix.append(grid_V.tolist())
            Xmatrix.append(grid_corx.tolist())
            Ymatrix.append(grid_cory.tolist())
            Zmatrix.append(grid_corz.tolist())

print(datetime.datetime.now())
