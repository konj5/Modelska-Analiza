import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta, gamma = args

    vdot[0] = -alpha * v[0] * v[1] - gamma
    vdot[1] = alpha * v[0] * v[1] - beta * v[1]
    vdot[2] = beta * v[1]

    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

startstate = np.array([0.99,0.01,0])
alpha = 1
beta = 0.1
gammas = np.linspace(0,0.5,100)

fig, axs = plt.subplots(1,3)


bmax, tmax, total = [], [], []
for i, gamma in enumerate(gammas):
    ts, vs = solve(startstate, f, [alpha, beta, gamma], 0, 10)

    #plt.plot(ts, vs[0,:])
    #plt.plot(ts, vs[1,:])
    #plt.plot(ts, vs[2,:])
    #plt.show()

    bmax.append(np.max(vs[1,:]))
    tmax.append(ts[np.argmax(vs[1,:])])
    total.append(vs[1,-1] + vs[2,-1])

axs[0].plot(gammas, bmax)
axs[1].plot(gammas, tmax)
axs[2].plot(gammas, total)


axs[0].set_title("Maksimum $B(t)$")
axs[1].set_title("Čas maksimuma $B(t)$")
axs[2].set_title("Delež populacije, \n ki je čez celo epidemijo \n kadarkoli okužen")

axs[0].set_xlabel("$\\gamma$")
axs[1].set_xlabel("$\\gamma$")
axs[2].set_xlabel("$\\gamma$")


plt.show()

