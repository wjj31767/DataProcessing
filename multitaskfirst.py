import numpy as np
from scipy import interpolate
import pandas as pd
import os
import time
import multiprocessing
import matplotlib.pyplot as plt
lower_T=1500000 # 温度被乘了1000

def GetMinSingleCore(files):
    dir_path = os.path.join(path, files[1])
    print(dir_path, files[0],time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test = pd.read_csv(dir_path,header=None,skiprows=[0,1],sep='\s+',)
    xy = pd.read_csv('xy.dat',header=None,skiprows=[0,1],sep='\s+')
    nn = pd.concat([xy[0]*100000,xy[1]*100000,test[1]*1000],axis=1)
    nn.columns = ['x', 'y', 'T']
    nn=nn.sort_values(by=['y','x'],ascending=True)
    nn.reset_index(drop=True, inplace=True)
    nn= pd.DataFrame(data=nn,dtype=np.int)
    tmp=nn[(nn['T']>=lower_T)].groupby('y')['x'].idxmin()
    TmpList4MIN = []
    for i in tmp:
        if nn.loc[i, :]['y'] == nn.loc[i - 1, :]['y']:
            # print(nn.loc[i,:],nn.loc[i-1,:])
            b = nn.loc[i, :]
            s = nn.loc[i - 1, :]
            # interpolating
            xd = (lower_T - s['T']) / (b['T'] - s['T']) * (b['x'] - s['x']) + s['x']
            TmpList4MIN.append([xd,b['y']])
    TmpList4MIN = pd.DataFrame(TmpList4MIN,columns=['x','y'])
    x = TmpList4MIN['y']
    y = TmpList4MIN['x']
    spl = interpolate.splrep(x, y)
    x2 = np.arange(x.min(),x.max(),0.1)
    y2 = interpolate.splev(x2, spl)
    xnew = y[:y.idxmin()]  # x
    ynew = x[:y.idxmin()]  # y
    MinValue = [[y2.min(),x2[y2.argmin()],np.degrees(np.arctan((xnew.max()-xnew.min())/(ynew.max()-ynew.min())))]]
    MinValuedir = os.path.join(tmppath,files[1])
    np.savetxt(MinValuedir,MinValue)
def GetMinMulti(path):
    filelist = os.listdir(path)
    pool = multiprocessing.Pool()
    for files in enumerate(filelist):
        pool.apply_async(GetMinSingleCore,(files,))
    pool.close()
    pool.join()

# megrge all the file under the tmp dir
def merge():
    tmp = pd.DataFrame()
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath,i)
        tmp = tmp.append(pd.read_csv(dir_path,header=None,sep='\s+',))
    np.savetxt('final',tmp)

if __name__=='__main__':
    paths = [r"a20/dataToThomas/a20"]
    # mkdir a file "a20tmp", if exits delete the files under the path
    tmppath = r'a20tmp'
    if not os.path.exists(tmppath):
        os.makedirs(tmppath)
    for i in os.listdir(tmppath):
        dir_path = os.path.join(tmppath,i)
        os.remove(dir_path)

    for path in paths:
        GetMinMulti(path)
        merge()
        for i in os.listdir(tmppath):
            dir_path = os.path.join(tmppath,i)
            os.remove(dir_path)
    os.removedirs('a20tmp')
    nn = pd.read_csv('final',header=None,skiprows=[0,1,2],sep='\s+',)
    plt.scatter(nn[0] / 100000, nn[1] / 100000, c=nn[2])
    plt.colorbar()
    plt.xlim((nn[0].min() - 1) / 100000, (nn[0].max() + 1) / 100000)
    plt.ylim((nn[1].min() - 10) / 100000, (nn[1].max() + 10) / 100000)
    plt.show()