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

import phaseportrait


def dz(z, l):
    return z * (1-l), l * (z-1)

SimplePendulum = phaseportrait.PhasePortrait2D(dz, [0, 5], MeshDim=40, Title='Simple pendulum', xlabel=r"$\Theta$", ylabel=r"$\dot{\Theta}$")
SimplePendulum.plot()

plt.show()