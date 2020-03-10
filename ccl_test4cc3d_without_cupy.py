import os
import datetime
import math
import time
import argparse
try:
    import cc3d
    import numpy as np
    import pandas as pd
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objs as go
    from tqdm import tqdm
except:
    os.system('pip install numpy')
    os.system('pip install pandas')
    os.system('pip install scipy')
    os.system('pip install matplotlib')
    os.system('pip install plotly')
    os.system('pip install connected-components-3d')
    os.system('pip install tqdm')
    import cc3d
    import numpy as np
    import pandas as pd
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objs as go
    from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=int,default=0,help='0 for all 3 steps: which can get the file and generate cc3d'
                                                     '1 for just generate 3d plot file from temp file, like cc3d.csv'
                                                     '2 for just generate histograph from temp file, like cc3d.csv')
parser.add_argument('--path', type=str, help='path of csv file',default='boxTest.csv')
parser.add_argument('--thresholdtype',type=int,default=1,help='1 for just 1 threshold'
                                                             '2 for 2 threshold')
parser.add_argument('--threshold',type=float,default=0.005,help="correspond to one threshold type")
parser.add_argument('--png',type=str,default='histgraph',help='path for result histograph path, will be seted its end as .png')
parser.add_argument('--html',type=str,default='3DPlot',help='path for result 3D plot path, will be seted its end as .html')
args = parser.parse_args()
if args.mode==0:
    LabelsInMatrix = []
    VolumnMatrix = []
    AlphaMatrix = []
    XMatrix = []
    YMatrix = []
    ZMatrix = []
    print(datetime.datetime.now())

    boxTextReadFromPandas = pd.read_csv(args.path)

    XGlobalMin = boxTextReadFromPandas['Points:1'].min()
    XGlobalMAX = boxTextReadFromPandas['Points:1'].max()
    YGlobalMIN = boxTextReadFromPandas['Points:2'].min()
    YGlobalMAX = boxTextReadFromPandas['Points:2'].max()

    print(XGlobalMin, XGlobalMAX, YGlobalMIN, YGlobalMAX)

    GridValue = 1e-3

    with open('cc3d.csv', 'w') as NewFile:
        NewFile.write("diameter,Volumn,MeanX,MeanY,MeanZ\n")

        n = 0
        # p.start(len(boxTextReadFromPandas.groupby("Points:0")))
        for num, DividedByPointZGroup in enumerate(tqdm(boxTextReadFromPandas.groupby("Points:0"))):
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
            if args.thresholdtype == 1:
                GridZ[GridZ < args.threshold] = 0
                GridZ[GridZ >= args.threshold] = 1
            GridZ = GridZ.astype(np.int32)

            # if num % 1000 == 0:
            #     print(datetime.datetime.now(), (num + 1) /len(boxTextReadFromPandas.groupby("Points:0")) )
            GridZ = np.asarray(GridZ)
            if np.sum(GridZ) == 0:
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
                    # calculate diameter
                    VolumnMatrix = np.asarray(VolumnMatrix)
                    AlphaMatrix = np.asarray(AlphaMatrix)
                    tmpMatrix = np.asarray(LabelsOutMatrix == segid)
                    sumtmpMatrix = np.multiply(VolumnMatrix, np.multiply(AlphaMatrix, tmpMatrix))
                    sumVolumn = np.sum(sumtmpMatrix)  # sum of Volumn
                    sumVolumndiameter = math.pow(sumVolumn * 6 / np.pi, 1 / 3)

                    # coordinate
                    XMatrix = np.asarray(XMatrix)
                    YMatrix = np.asarray(YMatrix)
                    ZMatrix = np.asarray(ZMatrix)
                    MeanX = np.sum(np.multiply(XMatrix, sumtmpMatrix)) / sumVolumn
                    MeanY = np.sum(np.multiply(YMatrix, sumtmpMatrix)) / sumVolumn
                    MeanZ = np.sum(np.multiply(ZMatrix, sumtmpMatrix)) / sumVolumn
                    # write into file
                    NewFile.write(str(sumVolumndiameter)
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
        #     if num % 1000 == 0:
        #         p.update(num + 1)
        # p.finish()
    print(datetime.datetime.now())
if args.mode==0 or args.mode==1:
    data = pd.read_csv('cc3d.csv')
    datasize = data["diameter"] * (10 ** 5) # grid 小的时候 5， 大的时候4
    datacolor = data["diameter"]
    fig1 = go.Scatter3d(x=data["MeanX"],
                        y=data["MeanY"],
                        z=data["MeanZ"],
                        marker=dict(opacity=0.9,
                                    reversescale=True,
                                    color=datacolor,
                                    size=datasize),
                        line=dict(width=0.02),
                        mode='markers')
    mylayout = go.Layout(scene=dict(xaxis=dict(title="x"),
                                    yaxis=dict(title="y"),
                                    zaxis=dict(title="z")), )
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                        auto_open=True,
                        filename=(args.html+'.html'))
if args.mode==0 or args.mode==2:
    drawforhist = np.genfromtxt('cc3d.csv',delimiter=',',skip_header=1)
    drawforhist = drawforhist[:,0]
    mi = drawforhist.min()
    mx = drawforhist.max()
    print(mi,mx)
    binnum = 100
    bin_edges = np.arange( mi ,mx,(mx-mi)/binnum)
    plt.hist(drawforhist,bins=bin_edges,color='red')
    plt.savefig(args.png+'.png')