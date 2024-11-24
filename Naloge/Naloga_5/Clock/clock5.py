import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm
from scipy.signal import argrelextrema

def fun(t,v, args):
    vdot = np.zeros_like(v)
    x, y, z, v, w = v
    
    alpha, lamb, c, d= args

    b = lamb * alpha
    a = alpha

    vdot[0] = -a*x - c*v*x + b*z*y + d*w*z
    vdot[1] = -b*z*y + c*v*x
    vdot[2] = -b*z*y - d*w*z
    vdot[3] = a*x - c*v*x
    vdot[4] = b*z*y - d*w*z
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/10000)
    return sol.t, sol.y



x0 = 0.6
y0 = 0
z0 = 1

a = 1

lamb = 10

names = ["x", "y", "z", "v", "w"]
ks = np.linspace(1,100,100)
ts, l1s, l2s, l3s, l4s, l5s = [], [], [], [], [], []
for j in tqdm(range(len(ks))):
    k = ks[j]
    c = k *  a
    d = k * lamb * a


    t,y = solve([x0,y0,z0,0,0], fun,[a, lamb, c, d],0,10)

    ts.append(t)
    l1s.append(y[0,:])
    l2s.append(y[1,:])
    l3s.append(y[2,:])
    l4s.append(y[3,:])
    l5s.append(y[4,:])



fig, axs = plt.subplots(2,3)

cmap = cm.get_cmap("viridis")
norm = colors.LogNorm(np.min(ks), np.max(ks))

axs[0,0].set_ylabel("x")
for i, y in enumerate(l1s):
    axs[0,0].plot(ts[0], y, c = cmap(norm(ks[i])))

axs[0,1].set_ylabel("y")
for i, y in enumerate(l2s):
    axs[0,1].plot(ts[1], y, c = cmap(norm(ks[i])))

axs[0,2].set_ylabel("z")
for i, y in enumerate(l3s):
    axs[0,2].plot(ts[2], y, c = cmap(norm(ks[i])))

axs[1,0].set_ylabel("v")
for i, y in enumerate(l4s):
    axs[1,0].plot(ts[3], y, c = cmap(norm(ks[i])))

axs[1,1].set_ylabel("w")
for i, y in enumerate(l5s):
    axs[1,1].plot(ts[4], y, c = cmap(norm(ks[i])))


axs[0,0].set_xlabel("t");axs[0,1].set_xlabel("t");axs[0,2].set_xlabel("t");axs[1,0].set_xlabel("t");axs[1,1].set_xlabel("t");

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = axs[1,1], label = "k")


###############################
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
a = 10

x0 = 0.6
y0 = 0
z0 = 1

t,y = solve([x0,y0,z0], fun,[a, Lamb],0,10)

names = ["x", "y", "z"]

for i in range(len(y[:,0])):
    axs[1,2].plot(t,y[i,:], label = f"{names[i]}")
    
axs[1,2].set_xlabel("t")
axs[1,2].legend()
axs[1,2].set_title("Približek stacionarnega \n stanja")
###############################

plt.show()


###### NARED COLORBAR GRAF KO GRE k -> infinity
