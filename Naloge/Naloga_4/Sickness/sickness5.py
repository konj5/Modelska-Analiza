import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fun(t,v, args):
    vdot = np.zeros_like(v)
    a, b, c, d, e, f, g  = args

    vdot[0] = -a * v[0] * v[1] - (b+e) * v[0] * v[2] + g * v[4]
    vdot[1] = a * v[0] * v[1] - c * v[1] + b * v[0] * v[2]
    vdot[2] = c * v[1] - (d+f) * v[2]
    vdot[3] = d * v[2]
    vdot[4] = f * v[2] + e * v[0] * v[2] - g * v[4]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

a=1
b=0.5
c=1
d=0
e=0.5
f=0.4
g=2

names = ["D", "$B_1$", "$B_2$", "M", "P"]
t,y = solve([10,1,0,0,0], fun,[a,b,c,d,e,f,g],0,15)
y = y / 11

for i in range(len(y[:,0])):
    plt.plot(t,y[i,:], label = f"{names[i]}")
plt.legend()
plt.show()