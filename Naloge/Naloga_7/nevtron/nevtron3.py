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

lp = alpha * d

N = 10**6

from nevtron1 import run_N as run_N_0

ls = np.linspace(0.1,6,40)
data = np.zeros((2,len(ls)))
for i in tqdm(range(len(ls)), leave = False):
    lp = ls[i]
    TR, endstates, bounces = run_N(N)
    data[0,i] = np.sum(TR)/N

    TR, bounces = run_N_0(N,lp)
    data[1,i] = np.sum(TR)/N

plt.plot(ls, data[0,:], label = "Izotropni model")
plt.plot(ls, data[1,:], label = "Poenostavljeni model", linestyle = "dashed")
plt.xlabel("$l_p$")
plt.ylabel("Transmisivnost")
plt.legend()
plt.show()
