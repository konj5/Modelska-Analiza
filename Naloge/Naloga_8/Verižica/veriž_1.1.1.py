import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


#Fixed edge points


def basic_H(h):
    E = 0
    for i in range(len(h)):
        E += alpha * h[i]*17/maxh
        if i != len(h)-1:
            E += 1/2 * ((h[i+1]*17/maxh-h[i]*17/maxh)**2 + 1)
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

def run_procedure(Nmin ):
    h = np.random.randint(0,maxh, 18); h[0] = maxh;  h[-1] = maxh
    E0 = basic_H(h)

    n = 0
    while(True):
        h1, E1 = poteza(h)

        print(f"{E1:0.2f}  {np.abs(E0-E1):0.2f}   n={n:0.0f}")

        if n > Nmin: break

        if np.abs(E0-E1) <= 0 and n > Nmin:
            break
        h = h1
        E0 = E1
        n += 1
    
    return h1, E1

minh = 0
#maxh = 17
maxh = 700

alpha = 0.5
kbT = 0.1


h,E = run_procedure(Nmin = 200000)

plt.scatter([i for i in range(len(h))], h/maxh*17, marker = "o", label = f"Rešitev")

from scipy.optimize import curve_fit

f = lambda x,a,C,d: a * np.cosh((x-d)/a) + C

params, trash = curve_fit(f, [i for i in range(len(h))], h/maxh*17, p0=[6,3.6,len(h)/2])

plt.plot(np.linspace(0,len(h),1000), f(np.linspace(0,len(h),1000), params[0], params[1], params[2]), color="r", linestyle = "dashed", label="Model")

print(E)
plt.title(f"$\\alpha = {alpha}$, $k_bT/k = 0.1$")
plt.legend()
plt.show()






