import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


path = "Naloge\\Naloga_7\\dodatna\\podatki\\"
letniki = [10,11,13,14]
naloge = [100 + i for i in range(1,14)]

data = np.empty((len(letniki), len(naloge), 30)); data[:,:,:] = np.nan

def toFloat(s):
    if s[0] == "-":
        sign = -1
        s = s[1:]
    else:
        sign = 1

    day, hour, minute = s.split(":")
    day = int(day); hour = int(hour); minute = int(minute)

    return sign * (24*day + hour + minute/60)
    
for i,letnik in enumerate(letniki):
    for j,naloga in enumerate(naloge):
        k = 0
        try:
            with open(path+"mod_tm{}_{}.dat".format(letnik,naloga)) as f:
                while(True):
                    s = f.readline()
                    #print(s); print(toFloat(s))
                    data[i,j,k] = toFloat(s)
                    k += 1
                
        except:
            print(path+"mod_tm{}_{}.dat".format(letnik,naloga))

"""letnik = 3
ts = np.linspace(-200,100, 10000)
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(1,13)

fig, ax = plt.subplots(1,1)

for i in range(len(naloge)):
    #print(data[letnik,i,:])
    #print([np.sum(data[letnik,i,:]<ts[j]) for j in range(len(ts))])

    ax.plot(ts, [np.sum(data[letnik,i,:]<ts[j]) for j in range(len(ts))], color = cmap(norm(i+1)))

plt.xlabel("Čas [h]")
plt.ylabel("Število oddanih nalog")
plt.colorbar(cm.ScalarMappable(norm,cmap), ax=ax, label = "Zaporedna številka naloge")
plt.title(f"Letnik 20{letniki[letnik]}")
plt.show()"""

from scipy.stats import kstest


statistic_table = np.zeros((len(letniki),len(naloge),len(letniki),len(naloge)))
p_table = np.zeros((len(letniki),len(naloge),len(letniki),len(naloge)))

for i in tqdm(range(len(letniki))):
    for j in tqdm(range(len(naloge)), leave=False):
        for k in tqdm(range(len(letniki)), leave=False):
            for l in tqdm(range(len(naloge)), leave=False):
                #print([i,j,k,l])
                temp =  kstest(data[i,j,:], data[k,l,:])
                statistic_table[i,j,k,l] = temp[0]
                p_table[i,j,k,l] = temp[1]









            
        
            


