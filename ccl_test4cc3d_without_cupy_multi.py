import datetime
import math
import multiprocessing

import cc3d
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import _multiprocessing
import os
GlobalList = [[]]
XGlobalMin = 0
XGlobalMAX = 0
YGlobalMIN = 0
YGlobalMAX = 0
ThreadValue = 0.01
GridValue = 1e-4
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
    n=0
    for DividedByPointZGroup in boxTextReadFromPandas.groupby("Points:0"):
        DividedByPointZGroup = DividedByPointZGroup[1].to_numpy()

        Points = DividedByPointZGroup[:, [6, 7]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        GridX, GridY = np.mgrid[XGlobalMin:XGlobalMAX:GridValue, YGlobalMIN:YGlobalMAX:GridValue]
        GridZ = griddata(Points, AlphaLiquid, (GridX, GridY), method='nearest')
        GridZ[GridZ < ThreadValue] = 0
        GridZ[GridZ >= ThreadValue] = 1
        GridZ = GridZ.astype(np.int32)
        if np.sum(GridZ) == 0:
            if(len(GlobalList[-1])!=0):
                GlobalList.append([])
            continue
        else:
            GlobalList[-1].append(DividedByPointZGroup.tolist())

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
        Tmp4SaveList.append([sumVolumnDelimeter,sumVolumn,MeanX,MeanY,MeanZ])
    np.savetxt(FilePath, Tmp4SaveList)
            # NewFile.write(str(sumVolumnDelimeter)
            #               + ',' + str(sumVolumn)
            #               + ',' + str(MeanX)
            #               + ',' + str(MeanY)
            #               + ',' + str(MeanZ)
            #               + '\n')
    print(i/len(GlobalList),datetime.datetime.now())
def ccl():
    pool = multiprocessing.Pool()
    for i in range(len(GlobalList)):
        pool.apply_async(cclonestep,(i,))
    pool.close()
    pool.join()

def merge():
    tmp = pd.DataFrame(columns=["delimeter","Volumn","MeanX","MeanY","MeanZ"])
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath,i)
        tmp = tmp.append(pd.read_csv(dir_path,header=None,sep='\s+',))
    np.savetxt('cc3d.csv',tmp)

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
    merge()
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath, i)
        os.remove(dir_path)
    os.removedirs(tmppath)
    print("end",datetime.datetime.now())