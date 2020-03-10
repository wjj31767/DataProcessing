import os
import datetime
import math
import time
import argparse

os.system('pip install scipy==1.4.1')
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
# https://github.com/seung-lab/connected-components-3d

'''
argparse part
'''
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0,
                    help='0(default) for all 3 steps: which can get the file and generate cc3d\n'
                         '1 for just generate 3d plot file from temp file, like cc3d.csv\n'
                         '2 for just generate histograph from temp file, like cc3d.csv')
parser.add_argument('--path', type=str, help='path of csv file, default is boxTest.csv', default='boxTest.csv')
parser.add_argument('--respath', type=str, default='cc3d',
                    help='path for result file path, default is cc3d, which will be seted its end with csv')
parser.add_argument('--png', type=str, default='histgraph',
                    help='path for result histograph path, will be seted its end as .png')
parser.add_argument('--html', type=str, default='3DPlot',
                    help='path for result 3D plot path, will be seted its end as .html')
parser.add_argument('--thresholdtype', type=int, default=1, help='1(default) for just 1 threshold\n'
                                                                 '2 for 2 threshold')
parser.add_argument('--threshold', type=float, default=0.005, help="correspond to one threshold type,default is 0.005")
parser.add_argument('--thresholdupper', type=float, default=0.012, help="correspond to top of two threshold type")
parser.add_argument('--thresholdlower', type=float, default=0.015, help="correspond to bottom of two threshold type")
parser.add_argument('--x', type=int, default=6, help="Point:1, default is column 6")
parser.add_argument('--y', type=int, default=7, help="Point:2, default is column 7")
parser.add_argument('--z', type=int, default=5, help="Point:0, default is column 5")
parser.add_argument('--v', type=int, default=0, help="Volumn, default is column 0")
parser.add_argument('--alpha', type=int, default=1, help="alphaLiquid, default is column 1")
parser.add_argument('--grid', type=float, default=1e-3, help="grid value, default is 1e-3")
parser.add_argument('--needrange', type=int, default=0, help='0(default) for all range of file\n'
                                                             '1 for expected range\n'
                                                             '2 for expected divided parts')
parser.add_argument('--parts', type=int, default=5, help='correspond to need range part 2:'
                                                         'it can divided the whole range into equal part')
parser.add_argument('--rangeupper', type=float, default=0.0, help='upper bound of expected range')
parser.add_argument('--rangelower', type=float, default=1.0, help='bottom bound of expected range')
parser.add_argument('--hbin', type=int, default=100, help='desired sum of bin, default is 100')
parser.add_argument('--pratio', type=int, default=5,
                    help='3d point scale ratio for 3d plot, default is 5, sometimes 4 is better')
args = parser.parse_args()
if args.needrange == 1:
    if args.rangeupper < args.rangelower:
        raise Exception("rangeupper smaller than rangelower", parser)


# main calculate part
def calculate(data, out_name):
    '''
    input: data is input csv data
            out_name is desired output name
    函数总体思路是按照每个切片的和来判断三维连通域是否中断，如果中断那么计算之前的连通域
    '''
    labelsin_matrix = []
    volumn_matrix = []
    alpha_matrix = []
    x_matrix = []
    y_matrix = []
    z_matrix = []
    print(datetime.datetime.now())
    with open(out_name + '.csv', 'w') as newfile:
        newfile.write("diameter,Volumn,MeanX,MeanY,MeanZ\n")
        for groupby_part in tqdm(data.groupby("Points:0")):
            groupby_part = groupby_part[1].to_numpy()
            # XYZ 的坐标对应Points： 0，1，2 Points [4,5] 对应的Points： 1 2
            Points = groupby_part[:, [args.x, args.y]]
            AlphaLiquid = groupby_part[:, args.alpha]
            v = groupby_part[:, args.v]
            x = groupby_part[:, args.x]
            y = groupby_part[:, args.y]
            z = groupby_part[:, args.z]
            gridx, gridy = np.mgrid[XGlobalMin:XGlobalMAX:args.grid, YGlobalMIN:YGlobalMAX:args.grid]
            gridz = griddata(Points, AlphaLiquid, (gridx, gridy), method='nearest')
            gridalpha = griddata(Points, AlphaLiquid, (gridx, gridy), method='nearest')
            gridvolumn = griddata(Points, v, (gridx, gridy), method='nearest')
            gridcoordinatex = griddata(Points, x, (gridx, gridy), method='nearest')
            gridcoordinatey = griddata(Points, y, (gridx, gridy), method='nearest')
            gridcoordinatez = griddata(Points, z, (gridx, gridy), method='nearest')
            if args.thresholdtype == 1:
                gridz[gridz < args.threshold] = 0
                gridz[gridz >= args.threshold] = 1
            elif args.thresholdtype == 2:
                gridz[args.thresholdlower <= gridz <= args.thresholdupper] = 1
                gridz[gridz < args.thresholdlower & gridz > args.thresholdupper] = 0
            gridz = gridz.astype(np.int32)
            gridz = np.asarray(gridz)
            if np.sum(gridz) == 0:
                labelsin_matrix = np.asarray(labelsin_matrix)
                if np.sum(labelsin_matrix) == 0:
                    labelsin_matrix = []
                    volumn_matrix = []
                    alpha_matrix = []
                    x_matrix = []
                    y_matrix = []
                    z_matrix = []
                    continue

                labelsout_matrix = cc3d.connected_components(labelsin_matrix)
                N = np.max(labelsout_matrix)
                for segid in range(1, N + 1):
                    # calculate diameter
                    volumn_matrix = np.asarray(volumn_matrix)
                    alpha_matrix = np.asarray(alpha_matrix)
                    tmpMatrix = np.asarray(labelsout_matrix == segid)
                    sumtmpMatrix = np.multiply(volumn_matrix, np.multiply(alpha_matrix, tmpMatrix))
                    sumVolumn = np.sum(sumtmpMatrix)  # sum of Volumn
                    sumVolumnDiameter = math.pow(sumVolumn * 6 / np.pi, 1 / 3)

                    # coordinate
                    x_matrix = np.asarray(x_matrix)
                    y_matrix = np.asarray(y_matrix)
                    z_matrix = np.asarray(z_matrix)
                    MeanX = np.sum(np.multiply(x_matrix, sumtmpMatrix)) / sumVolumn
                    MeanY = np.sum(np.multiply(y_matrix, sumtmpMatrix)) / sumVolumn
                    MeanZ = np.sum(np.multiply(z_matrix, sumtmpMatrix)) / sumVolumn
                    # write into file
                    newfile.write(str(sumVolumnDiameter)
                                  + ',' + str(sumVolumn)
                                  + ',' + str(MeanX)
                                  + ',' + str(MeanY)
                                  + ',' + str(MeanZ)
                                  + '\n')

                labelsout_matrix = []
                labelsin_matrix = []
                alpha_matrix = []
                volumn_matrix = []
                x_matrix = []
                y_matrix = []
                z_matrix = []
            else:
                labelsin_matrix.append(gridz.tolist())
                alpha_matrix.append(gridalpha.tolist())
                volumn_matrix.append(gridvolumn.tolist())
                x_matrix.append(gridcoordinatex.tolist())
                y_matrix.append(gridcoordinatey.tolist())
                z_matrix.append(gridcoordinatez.tolist())

    print(datetime.datetime.now())


if args.mode == 0:
    csv_file = pd.read_csv(args.path)
    XGlobalMin = csv_file['Points:1'].min()
    XGlobalMAX = csv_file['Points:1'].max()
    YGlobalMIN = csv_file['Points:2'].min()
    YGlobalMAX = csv_file['Points:2'].max()
    ZGlobalMIN = csv_file['Points:0'].min()
    ZGlobalMAX = csv_file['Points:0'].max()
    if args.needrange == 1:
        if args.rangeupper <= ZGlobalMIN or args.rangelower >= ZGlobalMAX:
            raise Exception('error setting rangeupper or rangelower,'
                            'rangeupper is %f,'
                            'rangelower is %f,'
                            'ZGlobalMin is %f,'
                            'ZGlobalMax is %f,' % (args.rangeupper, args.rangelower, ZGlobalMIN, ZGlobalMAX))
        csv_file = csv_file[
            (args.rangelower <= csv_file['Points:0']) & (csv_file["Points:0"] <= args.rangeupper)]
        calculate(csv_file, args.respath)
    elif args.needrange == 2:
        SubZ = ZGlobalMAX - ZGlobalMIN
        if SubZ <= 0:
            raise Exception("error Point:0", SubZ)
        for i in range(args.parts):
            calculate(csv_file[((i * SubZ / args.parts + ZGlobalMIN) <= csv_file['Points:0']) & (
                        csv_file["Points:0"] <= ((i + 1) * SubZ / args.parts + ZGlobalMIN))],
                      args.respath + '_' + str(i))
    else:
        calculate(csv_file, args.respath)
    print(XGlobalMin, XGlobalMAX, YGlobalMIN, YGlobalMAX, ZGlobalMIN, ZGlobalMAX)


# drawing the 3d plot
def draw3D(readPath, htmlName):
    '''
    input: readPath is result file name of caculate part
    output: 3d plot file with '.html'
    '''
    data = pd.read_csv(readPath + '.csv')
    datasize = data["diameter"] * (10 ** args.pratio)  # grid 小的时候 5， 大的时候4
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
    mylayout = go.Layout(scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
                                    xaxis=dict(title="x"),
                                    yaxis=dict(title="y"),
                                    zaxis=dict(title="z"),
                                    aspectmode='manual',  # this string can be 'data', 'cube', 'auto', 'manual'
                                    # a custom aspectratio is defined as follows:
                                    aspectratio=dict(x=1, y=1, z=0.95)
                                    ), )
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                        auto_open=True,
                        filename=(htmlName + '.html'))


if args.mode == 0 or args.mode == 1:
    if args.needrange == 2:
        for i in range(args.parts):
            draw3D(args.respath + '_' + str(i), args.html + '_' + str(i))
    else:
        draw3D(args.respath, args.html)


# drawing the histo graph
def drawHistograph(readPath, pngName):
    '''
    input: readPath is result file name of caculate part
    output: histo graph file end with '.png'
    '''
    drawforhist = np.genfromtxt(readPath + '.csv', delimiter=',', skip_header=1)
    drawforhist = drawforhist[:, 0]
    mi = drawforhist.min()
    mx = drawforhist.max()
    print(mi, mx)
    binnum = args.hbin
    bin_edges = np.arange(mi, mx, (mx - mi) / binnum)
    plt.hist(drawforhist, bins=bin_edges, color='red')
    plt.savefig(pngName + '.png')


if args.mode == 0 or args.mode == 2:
    if args.needrange == 2:
        for i in range(args.parts):
            drawHistograph(args.respath + '_' + str(i), args.png + '_' + str(i))
    else:
        drawHistograph(args.respath, args.png)
