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
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

p = 0.2
r = 0.5

t,y = solve([0.5,0.5], f,[r,p],0,200)
for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{i}")
plt.legend()
plt.show()