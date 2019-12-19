import datetime
import math
import cc3d
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

LabelsInMatrix = []
VolumnMatrix = []
AlphaMatrix = []
XMatrix = []
YMatrix = []
ZMatrix = []
print(datetime.datetime.now())

boxTextReadFromPandas = pd.read_csv('boxTest.csv')

XGlobalMin = boxTextReadFromPandas['Points:1'].min()
XGlobalMAX = boxTextReadFromPandas['Points:1'].max()
YGlobalMIN = boxTextReadFromPandas['Points:2'].min()
YGlobalMAX = boxTextReadFromPandas['Points:2'].max()

print(XGlobalMin, XGlobalMAX, YGlobalMIN, YGlobalMAX)

ThreadValue = 0.01
GridValue = 1e-4

with open('cc3d.csv', 'w') as NewFile:
    NewFile.write("delimeter,Volumn,MeanX,MeanY,MeanZ\n")

    n = 0

    for num, DividedByPointZGroup in enumerate(boxTextReadFromPandas.groupby("Points:0")):
        # print(num,DividedByPointZGroup[0])
        DividedByPointZGroup = DividedByPointZGroup[1].to_numpy()

        Points = DividedByPointZGroup[:, [6, 7]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        V = DividedByPointZGroup[:, 0]
        X = DividedByPointZGroup[:, 6]
        Y = DividedByPointZGroup[:, 7]
        Z = DividedByPointZGroup[:, 5]
        GridX, GridY = np.mgrid[XGlobalMin:XGlobalMAX:GridValue, YGlobalMIN:YGlobalMAX:GridValue]
        GridZ = griddata(Points, AlphaLiquid, (GridX, GridY), method='nearest')
        GridAlpha = griddata(Points, AlphaLiquid, (GridX, GridY), method='nearest')
        GridVolumn = griddata(Points, V, (GridX, GridY), method='nearest')
        GridCoordinateX = griddata(Points, X, (GridX, GridY), method='nearest')
        GridCoordinateY = griddata(Points, Y, (GridX, GridY), method='nearest')
        GridCoordinateZ = griddata(Points, Z, (GridX, GridY), method='nearest')
        GridZ[GridZ < ThreadValue] = 0
        GridZ[GridZ >= ThreadValue] = 1
        GridZ = GridZ.astype(np.int32)

        if num % 1000 == 0:
            print(datetime.datetime.now(), (num + 1) / 14297)
        GridZ = np.asarray(GridZ)
        if np.sum(GridZ) == 0:
            print(n)
            n += 1

            LabelsInMatrix = np.asarray(LabelsInMatrix)

            if np.sum(LabelsInMatrix) == 0:
                LabelsInMatrix = []
                VolumnMatrix = []
                AlphaMatrix = []
                XMatrix = []
                YMatrix = []
                ZMatrix = []
                continue

            # VolumnMatrix = np.power(np.asarray(VolumnMatrix),1/3)
            LabelsOutMatrix = cc3d.connected_components(LabelsInMatrix)
            N = np.max(LabelsOutMatrix)
            for segid in range(1, N + 1):
                # calculate delimeter
                VolumnMatrix = np.asarray(VolumnMatrix)
                AlphaMatrix = np.asarray(AlphaMatrix)
                tmpMatrix = np.asarray(LabelsOutMatrix == segid)
                sumtmpMatrix = np.multiply(VolumnMatrix, np.multiply(AlphaMatrix, tmpMatrix))
                sumVolumn = np.sum(sumtmpMatrix)  # sum of Volumn
                sumVolumnDelimeter = math.pow(sumVolumn * 6 / np.pi, 1 / 3)

                # coordinate
                XMatrix = np.asarray(XMatrix)
                YMatrix = np.asarray(YMatrix)
                ZMatrix = np.asarray(ZMatrix)
                MeanX = np.sum(np.multiply(XMatrix, sumtmpMatrix)) / sumVolumn
                MeanY = np.sum(np.multiply(YMatrix, sumtmpMatrix)) / sumVolumn
                MeanZ = np.sum(np.multiply(ZMatrix, sumtmpMatrix)) / sumVolumn
                # write into file
                NewFile.write(str(sumVolumnDelimeter)
                              + ',' + str(sumVolumn)
                              + ',' + str(MeanX)
                              + ',' + str(MeanY)
                              + ',' + str(MeanZ)
                              + '\n')

            LabelsOutMatrix = []
            LabelsInMatrix = []
            AlphaMatrix = []
            VolumnMatrix = []
            XMatrix = []
            YMatrix = []
            ZMatrix = []
        else:
            LabelsInMatrix.append(GridZ.tolist())
            AlphaMatrix.append(GridAlpha.tolist())
            VolumnMatrix.append(GridVolumn.tolist())
            XMatrix.append(GridCoordinateX.tolist())
            YMatrix.append(GridCoordinateY.tolist())
            ZMatrix.append(GridCoordinateZ.tolist())

print(datetime.datetime.now())
