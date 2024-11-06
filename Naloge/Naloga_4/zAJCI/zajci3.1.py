import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

import numpy as np
from scipy.signal import argrelextrema


def f(t,v, args):
    vdot = np.zeros_like(v)
    p, = args

    vdot[0] = p * v[0] * (1-v[1])
    vdot[1] = 1/p * v[1] * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y


z0s = np.linspace(0.01,10,100)
l0s = np.linspace(0.01,10,100)

p = 1
dts = np.zeros((len(z0s), len(l0s)))

xvals, yvals, colors = [], [], []

for i in tqdm(range(len(z0s))):
    for j in tqdm(range(len(l0s)), leave = False):
        z0 = z0s[i]; l0 = l0s[j];
        t,y = solve([z0,l0], f,[p],0,1000)
        x = y[1]
        y = y[0]

        extrema = argrelextrema(y, np.greater)
        extrema2 = argrelextrema(x, np.greater)

        if x[0] > y[0]:
            dts[i,j] = t[extrema2][1] - t[extrema][0]
        if x[0] < y[0]:
            dts[i,j] = t[extrema2][0] - t[extrema][1]

        xvals.append(z0); yvals.append(l0), colors.append(dts[i,j])

plt.scatter(xvals, yvals, s = 4, c = colors, cmap="viridis")
plt.colorbar(label = "zamik")
plt.xlabel("z")
plt.ylabel("l")
plt.show()





