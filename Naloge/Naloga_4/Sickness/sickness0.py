import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta, gamma = args

    vdot[0] = -alpha * v[0] * v[1] - gamma
    vdot[1] = alpha * v[0] * v[1] - beta * v[1]
    vdot[2] = beta * v[1] + gamma
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

t,y = solve([5,1,0], f,[1,1, 10],0,1)
for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{i}")
plt.legend()
plt.show()