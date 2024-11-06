import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

import numpy as np
from matplotlib import cm, colors


def f(t,v, args):
    vdot = np.zeros_like(v)
    p, = args

    vdot[0] = p * v[0] * (1-v[1])
    vdot[1] = 1/p * v[1] * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/400)
    return sol.t, sol.y


z0s = np.linspace(0.01,10,100)
l0s = np.linspace(0.01,10,100)

p = 1
dts = np.zeros((len(z0s), len(l0s)))

cmap = plt.get_cmap("viridis")
norm = colors.LogNorm(1,1000)
fig, ax = plt.subplots()
#ax, ax1 = axs

for i in tqdm(range(len(z0s))):
    for j in tqdm(range(len(l0s)), leave = False):
        z0 = z0s[i]; l0 = l0s[j];
        t,y = solve([z0,l0], f,[p],0,100)
        y = np.min(y)

        Nmin = 2; Nmax = 1000; vmin = y >= 1/Nmin; vmax = y >= 1/Nmax

        if vmin: ax.scatter([z0],[l0], s = 5, c = cmap(norm(2)))
        elif not vmax: pass
        elif vmin == False and vmax == True:
            while(Nmax - Nmin > 1):
                Nmid = (Nmax + Nmin)/2; vmid =  y >= 1/Nmid
                if vmid:
                    Nmax = Nmid
                    vmax = vmid
                else:
                    Nmin = Nmid
                    vmin = vmid

            Nmin = int(Nmin)
            while(vmin == False):
                Nmin += 1
                vmin = y >= 1/Nmin
            
            ax.scatter([z0],[l0], s = 5, c = cmap(norm(Nmin)))


fig.colorbar(cm.ScalarMappable(norm,cmap), ax=ax, label = "$N_{min}$")
ax.set_xlabel("z")
ax.set_ylabel("l")
plt.show()





