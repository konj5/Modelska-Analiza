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

    vdot[0] = -alpha * v[0] * v[1] + delta * v[2]
    vdot[1] = alpha * v[0] * v[1] - beta * v[1] - gamma * v[1]
    vdot[2] = beta * v[1] - delta * v[2]
    vdot[3] = gamma * v[1]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

def dN(x):
    sign = np.sign(x)
    return np.random.poisson(x)


dt = 0.001
tmax = 130


args = []


Dstart = 1000
Bstart = 10
Istart = 0
Mstart = 0

alpha = 0.001
beta = 0.1
gamma = 0.1
delta = 0

ts = np.arange(0,tmax,dt)
Ds = np.zeros_like(ts, dtype=np.int32); Ds[0] = Dstart
Bs = np.zeros_like(ts, dtype=np.int32); Bs[0] = Bstart
Is = np.zeros_like(ts, dtype=np.int32); Is[0] = Istart
Ms = np.zeros_like(ts, dtype=np.int32); Ms[0] = Mstart

for i in tqdm(range(1,len(ts))):

    """print(alpha*Bs[i-1]*Ds[i-1]*dt)
    print(alpha)
    print(Bs[i-1])
    print(Ds[i-1])
    print(dt)
    print("\n")"""

    A = dN(alpha*Bs[i-1]*Ds[i-1]*dt)
    B = dN(beta*Bs[i-1]*dt)
    C = dN(gamma*Bs[i-1]*dt)
    D = dN(delta*Is[i-1]*dt)


    dD = -A + D
    dB = A - B - C
    dI = B - D
    dM = C

    #print(dD + dB + dM + dI)

    Ds[i] = Ds[i-1] + dD if Ds[i-1] + dD > 0 else 0
    Bs[i] = Bs[i-1] + dB if Bs[i-1] + dB > 0 else 0
    Is[i] = Is[i-1] + dI if Is[i-1] + dI > 0 else 0
    Ms[i] = Ms[i-1] + dM if Ms[i-1] + dM > 0 else 0

    assert Ds[-1] >= 0
    assert Bs[-1] >= 0
    assert Is[-1] >= 0
    assert Ms[-1] >= 0

t,v = solve([Dstart, Bstart,Istart,Mstart], f, [alpha,beta,gamma,delta], 0, tmax)

print(Bs[-1])




plt.plot(ts, Ds, label = "Dovzetni")
plt.plot(ts, Bs, label = "Bolni")
plt.plot(ts, Is, label = "Imuni")
plt.plot(ts, Ms, label = "Mrtvi")

plt.plot(t, v[0,:], color = "black", linestyle = "dashed", label="Zvezna rešitev")
plt.plot(t, v[1,:], color = "black", linestyle = "dashed")
plt.plot(t, v[2,:], color = "black", linestyle = "dashed")
plt.plot(t, v[3,:], color = "black", linestyle = "dashed")

plt.xlabel("Čas")
plt.ylabel("Populacija")
plt.title(f"$\\alpha = {alpha}, \\beta = {beta}, \\gamma = {gamma}, \\delta = {delta}, \\Delta t = {dt}$")
plt.legend()
plt.tight_layout()
#plt.savefig(f"test dt {dt}.png")
plt.show()
