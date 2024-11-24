import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fun(t,v, args):
    vdot = np.zeros_like(v)
    alpha, = args

    vdot[0] = - v[0]**2 + alpha * v[0] * v[1]
    vdot[1] =  v[0]**2 - alpha * v[0] * v[1] -  v[1]
    vdot[2] =  v[1]
    vdot[3] =  v[1]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/100000)
    return sol.t, sol.y



alpha = 1000

fig, axs = plt.subplots(1,3)

from scipy.special import lambertw


def a(t, a0, alpha):
    
    return 1/alpha * 1/lambertw(1/alpha/a0 * np.exp((t+1/a0)/alpha))


for j, beta in enumerate([1]):

    a0 = 1/(alpha*beta)

    names = ["$a$", "$a^*$", "$b = c$"]
    t,y = solve([a0,0,0,0], fun,[alpha],0,10000)

    for i in range(len(names)):
        axs[i].plot(t,y[i,:], label = f"$\\beta = {beta}$")
        axs[i].set_title(names[i])
        
        axs[i].set_xlabel("$\\tau$")

        axs[i].legend()



axs[0].plot(t, [a(T,a0,alpha) for T in t], linestyle = "dashed")



plt.show()
