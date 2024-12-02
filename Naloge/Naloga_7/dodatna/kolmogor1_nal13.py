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
                """print(data[i,j,:])
                print(np.logical_not(np.isnan(data[i,j,:])))
                print(data[i,j,np.logical_not(np.isnan(data[i,j,:]))])"""

                temp =  kstest(data[i,j,np.logical_not(np.isnan(data[i,j,:]))], data[k,l,np.logical_not(np.isnan(data[k,l,:]))])
                statistic_table[i,j,k,l] = temp[0]
                p_table[i,j,k,l] = temp[1]



cmap = cm.get_cmap("gist_gray")
norm = colors.Normalize(0, 1)

#naloga = 3
for naloga in range(len(naloge)):

    naloga = 12

    data = p_table[:,naloga,:,naloga]

    p = len(letniki)-1

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(2.7,2)


    ax.imshow(np.delete(np.delete(data,1,axis=1),1,axis=0), cmap=cmap, norm=norm, extent=[0,p,0,p])

    # draw gridlines
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)

    ax.set_xticks([i for i in range(0,p,1)], [f"20{letniki[i]}" for i in range(0,p,1)])
    #plt.yticks([i for i in range(0,p,1)],  [f"20{letniki[i]}" for i in range(0,p,1)])

    letniki = [10,13,14,100]

    ax.yaxis.set(ticks=np.arange(0.5, len([f"20{letniki[i]}" for i in range(0,p,1)])), ticklabels=[f"20{letniki[i]}" for i in range(0,p,1)][::-1])
    ax.set_yticks(ticks=np.arange(0, len([f"20{letniki[i]}" for i in range(0,p,1)])), minor=True)
    ax.tick_params(axis='y', which='minor', length=10)
    ax.tick_params(axis='y', which='major', length=0)

    ax.xaxis.set(ticks=np.arange(0.5, len([f"20{letniki[i]}" for i in range(0,p,1)])), ticklabels=[f"20{letniki[i]}" for i in range(0,p,1)])
    ax.set_xticks(ticks=np.arange(0, len([f"20{letniki[i]}" for i in range(0,p,1)])), minor=True)

    fig.colorbar(cm.ScalarMappable(norm,cmap), ax = ax, label = "$p$")
    ax.set_title(f"Naloga 1{naloga+1:02d}")

    plt.savefig(f"kolgo nal{naloga+1:02d}.png")

    #plt.show()






            
        
            


