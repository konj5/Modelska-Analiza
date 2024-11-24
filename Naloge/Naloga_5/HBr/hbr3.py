import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors

def fun(t,v, args):
    vdot = np.zeros_like(v)
    x, y, z, u, v = v
    a, b, c, d, e = args


    vdot[0] = -a*x + b * u**2 - e*x*v
    vdot[1] = d*z*v - c*u*y
    vdot[2] = c*u*y - d*z*v + e*v*x
    vdot[3] = a*x - b*u**2 - c*u*y + d*z*v + e*v*x
    vdot[4] = c*u*y - d*z*v - e*v*x
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y


beta = 1

a=1
b=4
c=1
d=2
e=5



fig, axs = plt.subplots(1,2)


ts,vs = solve([1/(beta+1),beta/(beta+1),0,0,0], fun,[a,b,c,d,e],0,100)

axs[0].plot(ts, vs[0,:], label = "[Br$_2$]")
axs[0].plot(ts, vs[1,:], label = "[H$_2$]")
axs[0].plot(ts, vs[2,:], label = "[HBr]")

axs[1].plot(ts, vs[3,:], label = "[Br]")
axs[1].plot(ts, vs[4,:], label = "[H]")



axs[0].set_xlabel("t")
axs[1].set_xlabel("t")

axs[0].legend()
axs[1].legend()


plt.show()