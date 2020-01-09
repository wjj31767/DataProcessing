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
ThreadValue = 0.005
GridValue = 1.25e-5
def clip():
    CSVFile = pd.read_csv('6barFine.csv')
    CSVFile = CSVFile[(0.012<=CSVFile['Points:0'])&(CSVFile["Points:0"]<=0.015)]
    print(CSVFile)
    global XGlobalMAX
    global XGlobalMin
    global YGlobalMIN
    global YGlobalMAX
    XGlobalMin = CSVFile['Points:1'].min()
    XGlobalMAX = CSVFile['Points:1'].max()
    YGlobalMIN = CSVFile['Points:2'].min()
    YGlobalMAX = CSVFile['Points:2'].max()
    Groupby = CSVFile.groupby("Points:0")
    Groupnum = len(Groupby)
    for num,DividedByPointZGroup in enumerate(Groupby):
        DividedByPointZGroup = DividedByPointZGroup[1].to_numpy()

        Points = DividedByPointZGroup[:, [4, 5]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        GridX, GridY = np.mgrid[XGlobalMin:XGlobalMAX:GridValue, YGlobalMIN:YGlobalMAX:GridValue]
        GridZ = griddata(Points, AlphaLiquid, (GridX, GridY), method='nearest')
        GridZ[GridZ < ThreadValue] = 0
        GridZ[GridZ >= ThreadValue] = 1
        GridZ = GridZ.astype(np.int32)
        if num%1000==0:
            print(num/Groupnum,"\t",datetime.datetime.now())
        if np.sum(GridZ) == 0:
            if(len(GlobalList[-1])!=0):
                GlobalList.append([])
            continue
        else:
            GlobalList[-1].append(DividedByPointZGroup.tolist())
    del CSVFile
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
        Points = DividedByPointZGroup[:, [4, 5]]
        AlphaLiquid = DividedByPointZGroup[:, 1]
        V = DividedByPointZGroup[:, 0]
        X = DividedByPointZGroup[:, 4]
        Y = DividedByPointZGroup[:, 5]
        Z = DividedByPointZGroup[:, 3]
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
    print(i/len(GlobalList),"\t",datetime.datetime.now())
def ccl():
    pool = multiprocessing.Pool(15)
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
    np.savetxt('cc3dnew.csv',tmp,header="delimeter,volumn,meanx,meany,meanz",delimiter=",",comments="")

if __name__ == '__main__':
    tmppath = r'ccl4merge'
    if not os.path.exists(tmppath):
        os.makedirs(tmppath)
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath, i)
        os.remove(dir_path)
    print("start","\t",datetime.datetime.now())
    clip()
    print("clipend","\t",datetime.datetime.now())
    ccl()
    print("cclend","\t",datetime.datetime.now())
    merge()
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath, i)
        os.remove(dir_path)
    os.removedirs(tmppath)
    print("end","\t",datetime.datetime.now())
