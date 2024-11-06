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


z0s = np.linspace(0.01,10,10)
l0s = np.linspace(0.01,10,10)

p = 1
dts = np.zeros((len(z0s), len(l0s)))

cmap = plt.get_cmap("viridis")
norm = colors.LogNorm(1,1000)
fig, ax = plt.subplots()

l1x, l1y = [], []
l2x, l2y = [], []
gx, gy = [], []

for i in tqdm(range(len(z0s))):
    for j in tqdm(range(len(l0s)), leave = False):
        z0 = z0s[i]; l0 = l0s[j];
        t,y = solve([z0,l0], f,[p],0,100)
        miny = np.min(y)

        if miny < 1:
            l1x.append(z0); l1y.append(l0)
        elif miny < 2:
            l2x.append(z0); l2y.append(l0)
        else:
            gx.append(z0); gy.append(l0)
          
ax.scatter(l1x, l1y, s = 5, c = "red", label = "min < 1")
ax.scatter(l2x, l2y, s = 5, c = "orange", label = "min < 2")
ax.scatter(gx, gy, s = 5, c = "green", label = "min > 2")

ax.set_xlabel("z")
ax.set_ylabel("l")
plt.show()





