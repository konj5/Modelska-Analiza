import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


#Fixed edge points


def basic_H(h):
    E = 0
    for i in range(len(h)):
        E += alpha * h[i]
        if i != len(h)-1:
            E += 1/2 * ((h[i+1]-h[i])**2 + 1)
    return E

def get_pm_pos(h):
    while(True):
        pm = np.random.choice([-1,1])
        pos = np.random.choice(np.arange(1,len(h)-1))
        #print("ha")

        if h[pos] + pm <= maxh and h[pos] + pm >= minh:
            break
    return pm, pos

def poteza(h):
    E0 = basic_H(h)

    nmax = 100
    n = 0
    while(True):
        pm, pos = get_pm_pos(h)
        h1 = np.copy(h); h1[pos] += pm
        E1 = basic_H(h1)
        if E1 < E0:
            break
        if np.random.random() <= np.exp(-(E1-E0)/kbT):
            break
        n += 1
        if n > nmax:
            h1 = h; E1 = E0; break

    return h1, E1, n


def run_procedure(Nmin):
    h = h0
    E0 = basic_H(h)

    Es = [E0]
    Ntriess = [0]
    n = 0
    while(True):
        h1, E1, Ntries = poteza(h)
        Es.append(E1)
        Ntriess.append(Ntries)

        print(f"{E1}  {np.abs(E0-E1)}   n={n}")

        if np.abs(E0-E1) <= 0.5 and n > Nmin:
            break
        h = h1
        E0 = E1
        n += 1
    
    return h1, E1, Es, Ntriess

minh = 0
maxh = 17
#maxh = 10000

alpha = 1000

h0 = np.random.randint(0,maxh, 18); h0[0] = maxh;  h0[-1] = maxh


for kbT in [0.001, 0.1, 1, 2, 5]:
    h,E, Es, Ntriess = run_procedure(Nmin = 1000)

    #plt.plot(Es, label=f"$k_bT/k = {kbT}$")
    a = 10
    plt.plot([i for i in range(len(Ntriess))],[np.average(Ntriess[max(0,i-a):min(len(Ntriess)-1, i+a)]) for i in range(len(Ntriess))], label=f"$k_bT/k = {kbT}$")
    
    #print(E)
plt.title(f"$\\alpha = {alpha}$")
plt.xlabel("Korak")
#plt.ylabel("Energija")
plt.ylabel("Število zavrnjenih sprememb")
plt.legend()
plt.show()






