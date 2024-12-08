import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)


def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta, gamma, delta = args

    vdot[0] = -alpha * v[0] - beta * v[0] * v[1]
    vdot[1] = -gamma * v[1] + delta * v[0] * v[1]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

rng = np.random.default_rng()

def dN(factor, N, dt):
    return rng.poisson(factor*N*dt)  
    


dt = 0.001
tmax = 40

Zstart = 250
Lstart = 50

"""z_death = 4
z_birth = 5
z_ratio = 5/4

l_death = 5
l_birth = 4
l_ratio = 4/5"""

alpha = 1
beta = 1


L0 = 50
Z0 = 200

args = [-alpha, alpha/L0, beta, beta/Z0]


ts = np.arange(0,tmax,dt)

for _ in tqdm(range(5)):
    zs = np.zeros_like(ts, dtype=np.int32); zs[0] = Zstart
    ls = np.zeros_like(ts, dtype=np.int32); ls[0] = Lstart

    print(dN(1,0,1))
    for i in tqdm(range(1,len(ts)), leave=False):

        try:
            dZ = dN(5*alpha,zs[i-1],dt) - dN(4*alpha,zs[i-1],dt) - dN(alpha/L0,zs[i-1]*ls[i-1],dt)
            dL = dN(4*beta,ls[i-1],dt) - dN(5*beta,ls[i-1],dt) + dN(beta/Z0,zs[i-1]*ls[i-1],dt)
        except:
            try:
                dZ = dN(5*alpha,zs[i-1],dt) - dN(4*alpha,zs[i-1],dt) - dN(alpha/L0,zs[i-1]*ls[i-1],dt)
            except:
                dZ = 100

            try:
                dL = dN(4*beta,ls[i-1],dt) - dN(5*beta,ls[i-1],dt) + dN(beta/Z0,zs[i-1]*ls[i-1],dt)
            except:
                dL = 100

        zs[i] = zs[i-1] + dZ if zs[i-1] + dZ > 0 else 0
        ls[i] = ls[i-1] + dL if ls[i-1] + dL > 0 else 0

    
    plt.plot(zs, ls, zorder = 0, alpha = 0.5)

plt.scatter([Zstart],[Lstart], color = "black", zorder = 1)
plt.xlabel("Zajci")
plt.ylabel("Lisice")

plt.xlim(0,3000)
plt.ylim(0)


plt.show()