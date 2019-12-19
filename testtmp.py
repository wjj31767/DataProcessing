import numpy as np
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import math
nn = pd.read_csv('a20n/0.05.csv')
nn= pd.DataFrame(data=nn,dtype=np.int)
lower_T=1500000
tmp=nn[(nn['T']>=lower_T)].groupby('y')['x'].idxmin()
tt = pd.DataFrame(columns = ('x','y'))
for i in tmp:
    if nn.loc[i, :]['y'] == nn.loc[i - 1, :]['y']:
        # print(nn.loc[i,:],nn.loc[i-1,:])
        b = nn.loc[i, :]
        s = nn.loc[i - 1, :]
        # interpolating
        xd = (lower_T - s['T']) / (b['T'] - s['T']) * (b['x'] - s['x']) + s['x']
        tt=tt.append(pd.DataFrame({'x':[xd],'y':[b['y']]}),ignore_index=True)
# print(tt)
x = tt['y']
y = tt['x']
spl = interpolate.splrep(x, y)
x2 = np.arange(x.min(),x.max(),0.01)
y2 = interpolate.splev(x2, spl)
xnew = y[:y.idxmin()] #x
ynew = x[:y.idxmin()] #y
def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r
print((xnew.max()-xnew.min())//100)
print(xnew.max())
xtmp = []
angletmp = []
for i in range(int((xnew.max()-xnew.min())//((xnew.max()-xnew.min())//100))):
    xnewf = xnew[(xnew<xnew.min()+((xnew.max()-xnew.min())//100)*(i+1))&(xnew>xnew.min()+((xnew.max()-xnew.min())//100)*i)] #xnew - x
    ynewf = ynew[(xnew<xnew.min()+((xnew.max()-xnew.min())//100)*(i+1))&(xnew>xnew.min()+((xnew.max()-xnew.min())//100)*i)] #ynew - y
    #print(xnewf.values,ynewf.values)
    if len(xnewf)==0:
        continue
    a, b, r = linefit(xnewf.values, ynewf.values)
    xtmp += [(xnew.min()+((xnew.max()-xnew.min())//100)*i+xnew.min()+((xnew.max()-xnew.min())//100)*(i+1))/2]
    angletmp += [np.degrees(np.arctan(a))]

plt.plot(xtmp,angletmp)
plt.show()
# yder = interpolate.splev(xnew,spl,der=1)
# yderder = np.degrees(np.arctan(yder))
# print(xnew,ynew)
# print(yderder)
# print(yderder[abs(yderder-20).argmin()])
# plt.plot(xnew,yderder)
# # plt.plot(xnewf/100000,yder/100000)
# # plt.plot(xnewf/100000,xnewf/100000*a+b/100000)
# # plt.xlim(0, 0.032)
# # plt.ylim(0, 0.032)
# plt.show()
