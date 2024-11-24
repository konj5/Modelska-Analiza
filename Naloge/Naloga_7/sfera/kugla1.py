import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

def get_random_trio(lp):
    #r,costheta,s
    return [np.random.random()**(1/3), 2*np.random.random()-1, -lp*np.log(np.random.random())]

def run_one_particle(lp):
    r, costheta, s = get_random_trio(lp)
    return int(s > np.sqrt(r**2*(costheta**2-1)+R**2)-r*costheta)

def run_for_N(N,lp):
    escapees = 0
    for _ in range(N):
        escapees += run_one_particle(lp)

    return escapees/N
    
R = 1
lp = 1

#### Look at error

Ntries = 100
Ns = np.int32(10**np.linspace(1,6,40)[::-1])

STDs = np.zeros(len(Ns))
points = np.zeros((len(Ns), Ntries))
times = np.zeros(len(Ns))

import time
for i in tqdm(range(len(Ns))):
    N = Ns[i]
    stime = time.time()
    for j in tqdm(range(Ntries)):
        points[i,j] = run_for_N(N, lp)
    times[i] = (time.time()-stime)/Ntries

    STDs[i] = np.std(points[i,:])

from scipy.optimize import curve_fit
def f(x,a,b): return a / np.sqrt(x-b)
params, *_ = curve_fit(f, Ns, STDs, p0=[1,0])
a,b = params

fig, axs = plt.subplots(1,3)
ax1,ax2,ax3 = axs

for i in range(len(Ns)):
    ax1.scatter(Ntries * [Ns[i]], points[i,:], color = "gray", s = 3)

ax1.set_xlabel("N")
ax1.set_ylabel("Delež pobeglih fotonov")
ax1.set_xscale("log")

ax2.scatter(Ns,STDs, label = "Standardne deviacije")
ax2.plot(10**np.linspace(1,5,1000), f(10**np.linspace(1,5,1000), a,b), label="$\\frac{a}{\sqrt{N-b}}$", linestyle = "dashed", c = "red")
ax2.set_yscale("log")
ax2.set_xscale("log")

ax2.set_xlabel("N")
ax2.set_ylabel("Standardna deviacija")
ax2.legend()

ax3.plot(Ns,times)
#ax3.set_yscale("log")
#ax3.set_xscale("log")

ax3.set_xlabel("N")
ax3.set_ylabel("Čas računanja[s]")

plt.show()
        
