import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


######################

def fun0(t,v, args):
    vdot = np.zeros_like(v)
    alpha, = args

    vdot[0] = - v[0]**2 + alpha * v[0] * v[1]
    vdot[1] =  v[0]**2 - alpha * v[0] * v[1] -  v[1]
    vdot[2] =  v[1]
    vdot[3] =  v[1]
    return vdot


def solve0(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000000)
    return sol.t, sol.y



alpha = 1000

fig, axs = plt.subplots(1,2)

for j, beta in enumerate([10, 1, 0.1]):

    a0 = 1/(alpha*beta)

    names = ["$a$", "$a^*$", "$b = c$"]
    t,y = solve0([a0,0,0,0], fun0,[alpha],0,10000)

    

######################

def fun(t,v, args):
    vdot = np.zeros_like(v)
    alpha, = args

    vdot[0] = - v[0]**2 + alpha * v[0]**3 / (1+alpha*v[0])
    vdot[1] =  v[0]**2 / (1+alpha*v[0])
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000000)
    return sol.t, sol.y



alpha = 1

beta = 1

for i, beta in enumerate([10, 1, 0.1]):

    a0 = 1/(alpha*beta)

    names = ["Absolutna napaka $a$", "Absolutna napaka $b = c$"]
    t1,y1 = solve0([a0,0,0,0], fun0,[alpha],0,10000)

    t2,y2 = solve([a0,0], fun,[alpha],0,10000)


    for j in range(len(names)):
        axs[j].plot(t,np.abs(y1[j,:]-y2[j,:]), label = f"$\\beta = {beta}$")

        axs[j].set_title(names[j])
        axs[j].set_yscale("log")
        
        axs[j].set_xlabel("$\\tau$")

        axs[j].legend()

plt.show()

