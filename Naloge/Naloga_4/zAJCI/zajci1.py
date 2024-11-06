import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t,v, args):
    vdot = np.zeros_like(v)
    p, = args

    vdot[0] = p * v[0] * (1-v[1])
    vdot[1] = 1/p * v[1] * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

t,y = solve([100,1], f,[1],0,100)
n = ["Zajci", "Lisice"]
for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{n[i]}")
plt.legend()
plt.show()