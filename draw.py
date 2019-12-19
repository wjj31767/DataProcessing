import pandas as pd
import matplotlib.pyplot as plt
import sys
lower_T = 1500
lower_T *= 1000
paths = [r"a05", r"a10", r"a20", r"a40"]
def draw(path,lim):
    nn = pd.read_csv(path + 'txt', header=None, skiprows=[0, 1, 2], sep="\s+", )
    lim[0] = max(lim[0],nn[0].max())
    lim[1] = min(lim[1],nn[0].min())
    lim[2] = max(lim[2],nn[1].max())
    lim[3] = min(lim[3],nn[1].min())
    plt.scatter(nn[0] / 100000, nn[1] / 100000, c=nn[2],cmap = 'coolwarm_r')
    plt.text(x=nn[0].min()/100000, y=nn[1].min()/100000, s=path, fontsize=15)

if __name__ == '__main__':
    lim =[-sys.maxsize,sys.maxsize,-sys.maxsize,sys.maxsize]
    for path in paths:
        draw(path,lim)
    for path in paths:
        plt.text(0.0002, 0.007 + 0.05, path, ha='center', va='bottom', fontsize=7)
        break
    cb = plt.colorbar()
    cb.set_label('Angle/degree at Temperatur'+str(lower_T/1000)+'K')
    plt.xlim(1.7e-4,3e-4)
    plt.ylim(0.006,0.013)
    # plt.xlim((lim[1] - 1) / 100000, (lim[0] + 1) / 100000)
    # plt.ylim((lim[3] - 10) / 100000, (lim[2] + 10) / 100000)
    plt.grid(linestyle='-', color='0.5',linewidth=2)
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.savefig('final')

    plt.show()

