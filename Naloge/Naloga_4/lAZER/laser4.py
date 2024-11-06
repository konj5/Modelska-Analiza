import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from scipy.signal import argrelextrema

def f(t,v, args):
    vdot = np.zeros_like(v)
    r, p = args

    vdot[0] = r - p * v[0] * (v[1] + 1)
    vdot[1] = v[1] / p * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/10000)
    return sol.t, sol.y

p = 0.1
r = 0.3

fig, axs = plt.subplots(1,1)
ax1 = axs
from scipy.optimize import curve_fit

startstates = [[np.random.random()*2, np.random.random()*(r/p)] for _ in range(1)]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, state in enumerate(startstates):
    t,y = solve(state, f,[r,p],0,40)
    ax1.plot(t,np.sqrt((y[0,:]-1)**2+(y[1,:]-(r/p-1))**2), c = colors[i])

    Y = np.sqrt((y[0,:]-1)**2+(y[1,:]-(r/p-1))**2)
    extrema = np.array(argrelextrema(Y, np.greater))[0]


    period = np.average(t[extrema][1:] - t[extrema][:-1])

    relax_time, pcov, = curve_fit(f = lambda x, A, T: A * np.exp(-x/T), xdata = t, ydata = Y, p0 = [1,1])
    A = relax_time[0]
    relax_time = relax_time[1]

    ax1.plot(t,[A * np.exp(-x/relax_time) for x in t], c = "red", linestyle = "dashed")
    print(period)
    print(relax_time)

plt.show()