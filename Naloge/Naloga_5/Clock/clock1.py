import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors

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

Lamb = 100
a = 100

x0 = 0.6
y0 = 0
z0 = 1

t,y = solve([x0,y0,z0], fun,[a, Lamb],0,10)

names = ["[I]", "[I$_2$]", "[S$_2$O$_3$]"]

fig = plt.figure(figsize=(3, 3))

for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{names[i]}")
    
plt.xlabel("$\\tau$")
plt.legend()
plt.tight_layout()

plt.savefig(f"{Lamb} {a} {x0} {y0} {z0}.png")


plt.show()

