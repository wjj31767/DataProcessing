import pandas as pd
import time
import time

import pandas as pd

# txt中所有字符串读入data
if __name__ == '__main__':
    print('starttime',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    lists = [(0,0.0001),(0.0001,0.001),(0.001,0.1),(0.1,1.0)]

    def writelines(list,newtmp):
        tmp = []
        flag = False
        cnt = 0
        ncnt = 0
        for lines in zip(open('alpha.liquidMean','r'),open('V','r'),open('KlMean','r')):
            if lines[1][0] in '()':
                flag = not flag
                continue
            if flag:
                a = float(lines[0][:-1])
                v = float(lines[1][:-1])
                k = float(lines[2][:-1])
                if list[0] < a <= list[1]:
                    tmp.append([a,a*v,v*k])
                    cnt+=1
                    if cnt%10000000 == 0:
                        print(cnt,'included finished', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                else:
                    ncnt+=1
                    if ncnt%10000000 == 0:
                        print(ncnt,'not included finished', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('finished for loop',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        tmp = pd.DataFrame(tmp,columns= ['a','av','vk'])
        print(tmp)

        tmpcnt = 0
        for i in tmp.groupby('a'):
            tmpcnt+=1
            if tmpcnt%100000==0:
                print(tmpcnt,'finish final csv', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            newtmp.write(str(i[0])+' '+str(i[1]['av'].sum())+' '+str(i[1]['vk'].sum())+'\n')
        print('finish final csv', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    newtmp = open('out', 'w')
    for list in lists:
        writelines(list,newtmp)
    newtmp.close()
    print('finish write file csv', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))