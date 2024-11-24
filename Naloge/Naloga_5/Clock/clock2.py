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

ks = np.linspace(-50,50,10)

x0 = 0.5
y0 = 0
z0 = 1
a = 100
Lamb = 100

Ts = []

plt.figure(figsize=(5,3))

for i in tqdm(range(len(ks))):
    k = ks[i]
    t,y = solve([x0,y0,z0], fun,[a, Lamb+k],0,10)

    T = np.argmin(np.abs(y[0,:]-y[1,:]))    
    Ts.append(t[T])

plt.plot(ks, Ts)
plt.ylabel("Aktivacijski čas")
plt.xlabel("$\\Delta$")
plt.title("$x(0) = 0.5$, $y(0) = 0$, $z(0) = 1 $, $\\alpha = 100$, $\\lambda = 100+ \\Delta$")
plt.tight_layout()
plt.show()
