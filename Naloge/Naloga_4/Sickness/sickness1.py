import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta = args

    vdot[0] = -alpha * v[0] * v[1]
    vdot[1] = alpha * v[0] * v[1] - beta * v[1]
    vdot[2] = beta * v[1]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

N = 100
R = 1

startstates = np.zeros([N,3])
ratios = np.linspace(0,1,N)[::-1]
startstates[:,0] = R*ratios
startstates[:,1] = R*(1-ratios)

alpha = 1
beta = 0.1
args = alpha, beta


fig, axs = plt.subplots(1,3)

cmap = cm.get_cmap("Spectral")
norm = colors.Normalize(np.min(ratios), np.max(ratios))

startstates = startstates[::-1]

for i in range(len(startstates[:,0])):
    v0 = startstates[i,:]
    ts, vs = solve(v0, f, args, 0, 10)
    vs = vs / R

    axs[0].plot(ts, vs[0,:], color=cmap(norm(ratios[i])))
    axs[1].plot(ts, vs[1,:], color=cmap(norm(ratios[i])))
    axs[2].plot(ts, vs[2,:], color=cmap(norm(ratios[i])))

axs[0].set_ylim(0,1)
axs[1].set_ylim(0,1)
axs[2].set_ylim(0,1)

axs[0].set_title("D(t)")
axs[1].set_title("B(t)")
axs[2].set_title("I(t)")

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[-1], label = "D(0)")

plt.show()

