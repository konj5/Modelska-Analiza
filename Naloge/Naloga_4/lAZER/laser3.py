import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t,v, args):
    vdot = np.zeros_like(v)
    r, p = args

    vdot[0] = r - p * v[0] * (v[1] + 1)
    vdot[1] = v[1] / p * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/10000)
    return sol.t, sol.y

p = 2
r = 10

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs

startstates = [[np.random.random()*2, np.random.random()*(r/p)] for _ in range(4)]
startstates = [[1,0] for _ in range(1)]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, state in enumerate(startstates):
    t,y = solve(state, f,[r,p],0,40)
    ax1.plot(t,y[0,:], c = colors[i])
    ax1.plot(t,y[1,:], linestyle = "dashed", c = colors[i])
    ax2.plot(y[0,:], y[1,:], c = colors[i], label = f"$A(0) = {state[0]:0.1f}$ $F(0) = {state[1]:0.1f}$")

ax1.set_xlabel("$\\tau$")
ax2.set_xlabel("$A$")
ax2.set_ylabel("$F$")

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

A = Line2D([0],[1],color='black', label='A($\\tau$)')
F = Line2D([0],[1],color='black', label='F($\\tau$)', linestyle="dashed")

ax1.legend(handles=[A,F])
ax2.legend()

plt.show()