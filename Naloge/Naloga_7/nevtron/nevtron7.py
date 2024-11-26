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

lp = 0.5

N = 10**6

ls = np.linspace(0.2,4,40)

As, Bs, Cs = np.zeros(len(ls)),np.zeros(len(ls)),np.zeros(len(ls))

for i in tqdm(range(len(ls))):
    lp = ls[i]
    
    data, endstates, bounces = run_N(N)
    thetas = np.array([endstate[-1] for endstate in endstates])
    n, bins, patches = plt.hist(np.cos(thetas), np.linspace(-1,1,int(40)))
    
    n = n/N

    As[i] = n[-1]
    Bs[i] = (n[0] - n[20])/20
    Cs[i] = (n[-2] - n[20])/20

plt.show()

plt.plot(ls, As, label="a")
plt.plot(ls, Bs, label="b")
plt.plot(ls, Cs, label="c")
plt.xlabel("$\\frac{l_p}{R}$")
plt.legend()
plt.yscale("log")
plt.show()
