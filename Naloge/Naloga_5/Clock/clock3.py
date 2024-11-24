import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.signal import argrelextrema

def fun(t,v, args):
    x, y, z = v
    vdot = np.zeros_like(v)
    
    alpha, lamb = args

    vdot[0] = 2*alpha*(lamb * z**2 * y - x**2)
    vdot[1] = -alpha * (lamb * z**2 * y - x**2)
    vdot[2] = -2 * lamb * z**2 * y 
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

ks = np.linspace(0.1,2,100)

x0 = 0.5
y0 = 0
z0 = 1
a = 1

Ts = [[],[],[]]

plt.figure(figsize=(5,3))

for i in tqdm(range(len(ks))):
    for j, lamb in enumerate([1,10,100]):
        k = ks[i]
        t,y = solve([k,y0,z0], fun,[a, lamb],0,100)

        T = np.argmin(np.abs(y[0,:]-y[1,:]))    
        Ts[j].append(t[T])

plt.plot(ks, Ts[0], label = "$\\lambda = 1$")
plt.plot(ks, Ts[1], label = "$\\lambda = 10$")
plt.plot(ks, Ts[2], label = "$\\lambda = 100$")
plt.ylabel("Aktivacijski čas")
plt.xlabel("$x_0$")
plt.title("$x(0) = x_0$, $y(0) = 0$, $z(0) = 1 $, $\\alpha = 1$")
plt.tight_layout()
plt.legend()
plt.yscale("log")
plt.show()

