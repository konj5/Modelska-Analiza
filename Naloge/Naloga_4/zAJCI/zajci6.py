import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from scipy.signal import argrelextrema

import numpy as np
from matplotlib import cm, colors


def f(t,v, args):
    vdot = np.zeros_like(v)
    p, = args

    vdot[0] = p * v[0] * (1-v[1])
    vdot[1] = 1/p * v[1] * (v[0]-1)
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/4000)
    return sol.t, sol.y


v0s = [(np.random.random()*20, np.random.random()*20) for _ in range(5)]
v0s = [(10,10), (30,30), (40,40), (50,50)]
p = 1

for i, v0 in enumerate(v0s):
    t,y = solve(v0,f,[p], 0, 1000)
    
    plt.plot(y[0], y[1])
plt.xlabel("z")
plt.ylabel("l")
plt.show()

