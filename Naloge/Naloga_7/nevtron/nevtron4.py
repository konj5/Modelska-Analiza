import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

#Izotropni this time

def run_single():
    r = np.array([0,0,0], np.float64)
    dir = np.array([0,0,1], np.float64)
    costheta = 1
    phi = 0

    i = 0
    while(True):
        s = -lp * np.log(np.random.random())
        r += s * dir

        if r[2] < 0:
            return False, [r, phi, np.arccos(costheta)], i
        if r[2] > d:
            return True, [r, phi, np.arccos(costheta)], i
        
        i += 1

        phi = 2*np.pi*np.random.random()
        costheta = 2*np.random.random()-1
        sintheta = np.sqrt(1-costheta**2)

        dir = np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])
        
def run_N(N):
    data = []
    endstates = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T,pos,n = run_single()
        data.append(int(T))
        endstates.append(pos)
        bounces.append(n)
    
    return data, endstates, bounces


alpha = 1/2
beta = 1/2
d = 1

lp = 2

N = 10**6

data, endstates, bounces = run_N(N)

Nbs = np.arange(0,np.max(bounces)+1,1)
Bs, BsT, BsR = np.zeros_like(Nbs), np.zeros_like(Nbs), np.zeros_like(Nbs)
###Put into bins
bounces = np.array(bounces)
data = np.array(data)
for i in tqdm(range(len(Nbs))):
    Bs[i] = np.sum(np.int32(bounces == i))
    BsT[i] = np.sum(np.int32(bounces == i) * np.int32(data == 1))
    BsR[i] = Bs[i] - BsT[i] 

plt.bar(Nbs-0.3, Bs, label = "Skupaj", width = 0.3)
plt.bar(Nbs, BsT, label = "Prepuščen", width = 0.3)
plt.bar(Nbs+0.3, BsR, label = "Odbit", width = 0.3)
plt.xticks(Nbs[::10])
plt.legend()

plt.xlabel("Število odbojev")
plt.ylabel("Število nevtronov")

plt.title(f"$N = 10^6$, $d = {d:0.1f}$, $l_p = {lp:0.3f}$, $T = {np.sum(data)/N:0.3f}$, $R = {1-np.sum(data)/N:0.3f}$")


#plt.yscale("log")
plt.show()

plt.bar(Nbs-0.3, Bs, label = "Skupaj", width = 0.3)
plt.bar(Nbs, BsT, label = "Prepuščen", width = 0.3)
plt.bar(Nbs+0.3, BsR, label = "Odbit", width = 0.3)
plt.xticks(Nbs[::4])
plt.legend()

plt.xlabel("Število odbojev")
plt.ylabel("Število nevtronov")

plt.title(f"$N = 10^6$, $d = {d:0.1f}$, $l_p = {lp:0.1f}$, $T = {np.sum(data)/N:0.3f}$, $R = {1-np.sum(data)/N:0.3f}$")

plt.yscale("log")
plt.show()