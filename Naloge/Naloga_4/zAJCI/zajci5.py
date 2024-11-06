import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from scipy.signal import argrelextrema

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


xs = np.linspace(100,1000,10)
ps = np.linspace(0.01,2,10)


xvals, yvals, colors = [], [], []

for i in tqdm(range(len(xs))):
    for j in tqdm(range(len(ps)), leave = False):

        z0 = 1; l0 = xs[i]; p = ps[j]

        t,y = solve([z0,l0], f,[p],0,1000)
        y = y[0]
        extrema = argrelextrema(y, np.greater)

        xvals.append(l0); yvals.append(p), colors.append(np.average(t[extrema][1:] - t[extrema][:-1]))

plt.scatter(xvals, yvals, s = 4, c = colors, cmap="viridis")
plt.colorbar(label = "perioda")
plt.xlabel("x")
plt.ylabel("p")
plt.title("Perioda rešitev z začetnimi stanji (1,x)")
plt.show()
