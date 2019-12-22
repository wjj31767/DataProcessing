import datetime
import math
import multiprocessing

import cc3d
import numpy as np
import pandas as pd
from numba import jit
from scipy.interpolate import griddata
import _multiprocessing
import os
import numba
GlobalList = [[]]
XGlobalMin = 0
XGlobalMAX = 0
YGlobalMIN = 0
YGlobalMAX = 0
ThreadValue = 0.01
GridValue = 1e-4
@jit
def npsum(x):
    return np.sum(x)
@jit
def npmultiply(a,b):
    return np.multiply(a,b)
def clip():
    boxTextReadFromPandas = pd.read_csv('boxTest.csv')
    global XGlobalMAX
    global XGlobalMin
    global YGlobalMIN
    global YGlobalMAX
    XGlobalMin = boxTextReadFromPandas['Points:1'].min()
    XGlobalMAX = boxTextReadFromPandas['Points:1'].max()
    YGlobalMIN = boxTextReadFromPandas['Points:2'].min()
    YGlobalMAX = boxTextReadFromPandas['Points:2'].max()
    for DividedByPointZGroup in boxTextReadFromPandas.groupby("Points:0"):
        DividedByPointZGroup = DividedByPointZGroup[1].to_numpy()

        Points = DividedByPointZGroup[:, [6, 7]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        GridX, GridY = np.mgrid[XGlobalMin:XGlobalMAX:GridValue, YGlobalMIN:YGlobalMAX:GridValue]
        GridZ = griddata(Points, AlphaLiquid, (GridX, GridY), method='nearest')
        GridZ[GridZ < ThreadValue] = 0
        GridZ[GridZ >= ThreadValue] = 1
        GridZ = GridZ.astype(np.int32)
        if npsum(GridZ) == 0:
            if(len(GlobalList[-1])!=0):
                GlobalList.append([])
            continue
        else:
            GlobalList[-1].append(DividedByPointZGroup.tolist())
    del boxTextReadFromPandas
def cclonestep(i):
    LabelsInMatrix = []
    VolumnMatrix = []
    AlphaMatrix = []
    XMatrix = []
    YMatrix = []
    ZMatrix = []
    FilePath = os.path.join(tmppath,str(i)+".csv")
    global GlobalList
    for DividedByPointZGroup in GlobalList[i]:
        DividedByPointZGroup = np.asarray(DividedByPointZGroup)
        Points = DividedByPointZGroup[:, [6, 7]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        V = DividedByPointZGroup[:, 0]
        X = DividedByPointZGroup[:, 6]
        Y = DividedByPointZGroup[:, 7]
        Z = DividedByPointZGroup[:, 5]
        global XGlobalMAX
        global XGlobalMin
        global YGlobalMIN
        global YGlobalMAX
        global GridValue
        global ThreadValue
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
        LabelsInMatrix.append(GridZ.tolist())
        AlphaMatrix.append(GridAlpha.tolist())
        VolumnMatrix.append(GridVolumn.tolist())
        XMatrix.append(GridCoordinateX.tolist())
        YMatrix.append(GridCoordinateY.tolist())
        ZMatrix.append(GridCoordinateZ.tolist())
    LabelsInMatrix = np.asarray(LabelsInMatrix)
    LabelsOutMatrix = cc3d.connected_components(LabelsInMatrix)
    N = np.max(LabelsOutMatrix)
    Tmp4SaveList = []
    for segid in range(1, N + 1):
        # calculate delimeter
        VolumnMatrix = np.asarray(VolumnMatrix)
        AlphaMatrix = np.asarray(AlphaMatrix)
        tmpMatrix = np.asarray(LabelsOutMatrix == segid)
        sumtmpMatrix = npmultiply(VolumnMatrix, npmultiply(AlphaMatrix, tmpMatrix))
        sumVolumn = npsum(sumtmpMatrix)  # sum of Volumn
        sumVolumnDelimeter = math.pow(sumVolumn * 6 / np.pi, 1 / 3)

        # coordinate
        XMatrix = np.asarray(XMatrix)
        YMatrix = np.asarray(YMatrix)
        ZMatrix = np.asarray(ZMatrix)
        MeanX = npsum(npmultiply(XMatrix, sumtmpMatrix)) / sumVolumn
        MeanY = npsum(npmultiply(YMatrix, sumtmpMatrix)) / sumVolumn
        MeanZ = npsum(npmultiply(ZMatrix, sumtmpMatrix)) / sumVolumn
        # write into file
        Tmp4SaveList.append([sumVolumnDelimeter,sumVolumn,MeanX,MeanY,MeanZ])
    np.savetxt(FilePath, Tmp4SaveList)
    print(i/len(GlobalList),datetime.datetime.now())
def ccl():
    pool = multiprocessing.Pool()
    for i in range(len(GlobalList)):
        pool.apply_async(cclonestep,(i,))
    pool.close()
    pool.join()

def merge():
    tmp = []
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath,i)
        t = np.genfromtxt(dir_path)
        if len(t.shape)!=2:
            t=[t.tolist()]
        else:
            t=t.tolist()
        tmp +=t
    np.savetxt('cc3d.csv',tmp,header="delimeter,volumn,meanx,meany,meanz",delimiter=",",comments="")


if __name__ == '__main__':
    tmppath = r'ccl4merge'
    if not os.path.exists(tmppath):
        os.makedirs(tmppath)
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath, i)
        os.remove(dir_path)
    print("start",datetime.datetime.now())

    clip()

    print("clipend",datetime.datetime.now())
    ccl()
    print("cclend",datetime.datetime.now())
    merge()
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath, i)
        os.remove(dir_path)
    os.removedirs(tmppath)
    print("end",datetime.datetime.now())