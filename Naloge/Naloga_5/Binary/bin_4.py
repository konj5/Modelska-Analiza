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
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/30000)
    return sol.t, sol.y





    

######################

def fun(t,v, args):
    vdot = np.zeros_like(v)
    alpha, = args

    vdot[0] = - v[0]**2 + alpha * v[0]**3 / (1+alpha*v[0])
    vdot[1] =  v[0]**2 / (1+alpha*v[0])
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/30000)
    return sol.t, sol.y



fig, axs = plt.subplots(1,2)

alpha = 1000

beta = 1

from scipy.special import lambertw

def a(t, a0, alpha):
    
    return 1/alpha * 1/lambertw(1/alpha/a0 * np.exp((t+1/a0)/alpha))

for i, beta in enumerate([10, 1, 0.1]):

    a0 = 1/(alpha*beta)

    names = ["Relativna napaka $a$", "Relativna napaka $b = c$"]
    t1,y1 = solve0([a0,0,0,0], fun0,[alpha],0,10000); y1[1,:] = y1[2,:]
    t2,y2 = solve([a0,0], fun,[alpha],0,10000)


    t = t2
    y = y2

    ya = np.zeros_like(y)
    ya[0,:] = np.array([a(T,a0,alpha) for T in t])

    ya[1,:] = np.array([np.trapezoid(y = np.pow(ya[0,:_+1], 2)/(1+alpha*ya[0,:_+1]), x = t[:_+1]) for _ in range(len(t))])

    
    """plt.plot(t, y[0,:], c = "r")
    plt.plot(t, y[1,:], c = "b")

    plt.plot(t1, y1[0,:], c = "r", linestyle = "dashed")
    plt.plot(t1, y1[1,:], c = "b", linestyle = "dashed")

    plt.show()"""


    for j in range(len(names)):
        axs[j].plot(t,np.abs((ya[j,:len(y[0,:])]-y[j,:])/(ya[j,:len(y[0,:])]+y[j,:])*2), label = f"$\\beta = {beta}$")

        axs[j].set_title(names[j])
        axs[j].set_yscale("log")
        
        axs[j].set_xlabel("$\\tau$")

        axs[j].legend()



plt.show()

