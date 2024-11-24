import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors

def fun(t,v, args):
    x, y, z = v
    vdot = np.zeros_like(v)
    k, m, = args

    zdot = k * np.sqrt(x) * y / (m + z/x)

    vdot[0] = -zdot/2
    vdot[1] = -zdot/2
    vdot[2] = zdot
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y


beta = 100

m = 2.5


ks = np.linspace(0,4,100)

fig, axs = plt.subplots(1,3)

cmap = cm.get_cmap("Spectral")
norm = colors.Normalize(np.min(ks), np.max(ks))

for i, k in enumerate(ks):
    ts,vs = solve([1/(beta+1),beta/(beta+1),1], fun,[k,m],0,100)

    axs[0].plot(ts, vs[0,:], color=cmap(norm(k)))
    axs[1].plot(ts, vs[1,:], color=cmap(norm(k)))
    axs[2].plot(ts, vs[2,:], color=cmap(norm(k)))

axs[0].set_title("Br$_2$(t)")
axs[1].set_title("H$_2$(t)")
axs[2].set_title("HBr(t)")

axs[0].set_xlabel("t")
axs[1].set_xlabel("t")
axs[2].set_xlabel("t")

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[-1], label = "k")


plt.show()