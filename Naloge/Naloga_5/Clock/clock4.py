import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.signal import argrelextrema

def fun(t,v, args):
    vdot = np.zeros_like(v)
    x, y, z, v, w = v
    
    alpha, lamb, c, d= args

    b = lamb * alpha
    a = alpha

    vdot[0] = -a*x - c*v*x + b*z*y + d*w*z
    vdot[1] = -b*z*y + c*v*x
    vdot[2] = -b*z*y - d*w*z
    vdot[3] = a*x - c*v*x
    vdot[4] = b*z*y - d*w*z
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/10000)
    return sol.t, sol.y



x0 = 0.6
y0 = 0
z0 = 1

a = 1

lamb = 10
k = 1

c = k *  a
d = k * lamb * a

Ts = [[],[],[]]

t,y = solve([x0,y0,z0,0,0], fun,[a, lamb, c, d],0,10)

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs

names = ["x", "y", "z", "v", "w"]
for i in range(3):
    ax1.plot(t,y[i,:], label = names[i])
for i in range(3,5):
    ax2.plot(t,y[i,:], label = names[i])

ax1.set_xlabel("t"); ax2.set_xlabel("t")
ax1.legend(); ax2.legend()

plt.show()


###### NARED COLORBAR GRAF KO GRE k -> infinity
