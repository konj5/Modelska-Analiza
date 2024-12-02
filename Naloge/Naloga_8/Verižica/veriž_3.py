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

    nmax = 10
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

    return h1, E1

def run_procedure(Nmin, Nmax):
    h = np.random.randint(0,maxh, 18); h[0] = maxh;  h[-1] = maxh
    E0 = basic_H(h)

    n = 0
    while(True):
        h1, E1 = poteza(h)

        #print(f"{E1}  {np.abs(E0-E1)}   n={n}")

        if (np.abs(E0-E1) < 0.5 and n > Nmin) or n > Nmax:
            break
        h = h1
        E0 = E1
        n += 1
    
    return h1, E1

minh = 0
maxh = 17
#maxh = 10000

kbTs = np.linspace(0.1, 5, 10)
alphas = np.linspace(0.1,3,4)

for j in tqdm(range(len(alphas))):
    alpha = alphas[j]
    Es = []
    ESTDs= []

    for i in tqdm(range(len(kbTs)), leave=False):
        kbT = kbTs[i]
        Estemp = []
        for _ in tqdm(range(20), leave = False):
            h,E = run_procedure(Nmin = 1000, Nmax = 2000)

            Estemp.append(E)

        Es.append(np.average(Estemp))
        ESTDs.append(np.std(Estemp))

    Es = np.array(Es); ESTDs = np.array(ESTDs)

    plt.plot(kbTs, Es, label = f"$\\alpha ={alpha:0.2f}$")
    #plt.errorbar(kbTs, Es, ESTDs, label = f"$\\alpha ={alpha:0.2f}$", capsize=5)
    plt.fill_between(kbTs, Es+ESTDs, Es-ESTDs, alpha = 0.4)

plt.legend()
plt.ylabel("$E$")
plt.xlabel("$k_bT/k$")
plt.title("$E(T)$ z napako")
plt.show()
        






