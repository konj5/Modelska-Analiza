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


data, endstates, bounces = run_N(N)

thetas = np.array([endstate[-1] for endstate in endstates])

#plt.hist(thetas, np.linspace(0,np.pi,int(np.pi*40)))
plt.hist(np.cos(thetas), np.linspace(-1,1,int(40)))
plt.legend()

plt.xlabel("$\\cos\\theta$")
plt.ylabel("Število nevtronov")

#plt.xticks(np.array([i/4 for i in range(5)])*np.pi,
#    ["$0$","$\\frac{\\pi}{4}$","$\\frac{\\pi}{2}$","$\\frac{3\\pi}{4}$","$\\pi$"])

plt.title(f"$N = 10^6$, $d = {d:0.1f}$, $l_p = {lp:0.3f}$")


#plt.yscale("log")
plt.show()