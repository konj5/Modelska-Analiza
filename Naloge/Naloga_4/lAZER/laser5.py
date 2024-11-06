import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

def f(t,v, args):
    vdot = np.zeros_like(v)
    r, p = args

    vdot[0] = r - p * v[0] * (v[1] + 1)
    vdot[1] = v[1] / p * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/10000)
    return sol.t, sol.y

ps = np.linspace(0.01,2,100)
rs = np.linspace(0.01,2,100)

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs
state = [1,1]

xdata, ydata, pdata, tdata  = [], [], [], []

for i in tqdm(range(len((ps)))):
    for j in tqdm(range(len((rs))), leave=False):
        p = ps[i]; r = rs[j]

        if p > r: continue


        t,y = solve(state, f,[r,p],0,400)

        Y = np.sqrt((y[0,:]-1)**2+(y[1,:]-(r/p-1))**2)
        extrema = np.array(argrelextrema(Y, np.greater))[0]

        period = np.average(t[extrema][1:] - t[extrema][:-1])

        relax_time, pcov, = curve_fit(f = lambda x, A, T: A * np.exp(-x/T), xdata = t, ydata = Y, p0 = [1,1]);A = relax_time[0];relax_time = relax_time[1]

        xdata.append(p)
        ydata.append(r)
        pdata.append(period)
        tdata.append(relax_time)

s = ax1.scatter(xdata, ydata, s = 5, c = pdata, norm = "log", cmap="viridis")
fig.colorbar(s, ax = ax1, label = "perioda")
ax1.set_xlabel("p")
ax1.set_ylabel("r")

ax2.scatter(xdata, ydata, s = 5, c = tdata, norm = "log", cmap="viridis")
fig.colorbar(s, ax = ax2, label = "relaksacijski čas")
ax2.set_xlabel("p")
ax2.set_ylabel("r")

plt.show()