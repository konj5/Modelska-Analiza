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
tmax = 80

Zstart = 200
Lstart = 50

"""z_death = 4
z_birth = 5
z_ratio = 5/4

l_death = 5
l_birth = 4
l_ratio = 4/5"""

alpha = 1
beta = 1


Z0 = 200
L0 = 50

args = [-alpha, alpha/L0, beta, beta/Z0]


ts = np.arange(0,tmax,dt)

alphas = np.linspace(0.1,2,10)
betas = np.linspace(0.1,2,10)
death_z = np.zeros((len(alphas), len(betas)))
death_l = np.zeros((len(alphas), len(betas)))

for u in tqdm(range(len(alphas))):
    alpha = alphas[u]
    for v in tqdm(range(len(betas)), leave=False):
        beta = betas[v]
        death_z_temp = []
        death_l_temp = []

        for _ in tqdm(range(40), leave=False):
            zs = np.zeros_like(ts, dtype=np.int32); zs[0] = Zstart
            ls = np.zeros_like(ts, dtype=np.int32); ls[0] = Lstart

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

            c1 = False
            c2 = False
            for i in range(len(ts)): 
                if zs[i] == 0 and not c1:
                    death_z_temp.append(ts[i])
                    c1 = True

                if ls[i] == 0 and not c2:
                    death_l_temp.append(ts[i])
                    c2 = True
                elif c1 and c2:
                    break
        
        death_z[u,v] = np.average(death_z_temp)
        death_l[u,v] = np.average(death_l_temp)

fig, axs = plt.subplots(1,2); ax1,ax2=axs

cmap = cm.get_cmap("viridis")
norm = colors.Normalize(0, tmax)

ax1.imshow(death_z, cmap=cmap, norm=norm, origin="lower", extent=[alphas[0], alphas[-1], betas[0], betas[-1]])
#fig.colorbar(cm.ScalarMappable(norm,cmap), label="Čas izumrtja", ax=ax1)
ax1.set_xlabel("$\\alpha$")
ax1.set_ylabel("$\\beta$")
ax1.set_title("Zajci")

#cmap = cm.get_cmap("viridis")
#norm = colors.Normalize(0, tmax)

print(death_z)

ax2.imshow(death_z, cmap=cmap, norm=norm, origin="lower", extent=[alphas[0], alphas[-1], betas[0], betas[-1]])
fig.colorbar(cm.ScalarMappable(norm,cmap), label="Čas izumrtja", ax=ax2)
ax2.set_xlabel("$\\alpha$")
ax2.set_ylabel("$\\beta$")
ax2.set_title("Lisice")

print(death_l)

plt.show()


