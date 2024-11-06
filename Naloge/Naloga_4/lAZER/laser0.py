import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, B1, beta, B2, R = args

    vdot[0] = -alpha * v[0] - B1 * v[0] * v[1]
    vdot[1] = -beta * v[1] + B2 * v[0] * v[1] + R
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

t,y = solve([10,10], f,[1,1,1,1,10],0,1)
for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{i}")
plt.legend()
plt.show()