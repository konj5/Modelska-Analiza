import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re


def exact_sol(t,beta,N0):
    return N0 * np.exp(-beta*t)

def stoch_solve(dt, N0, beta):
    ts = [0]
    Ns = [N0]

    while Ns[-1] > 0:

        dN = -np.random.poisson(beta * Ns[-1] * dt)

        ts.append(ts[-1] + dt)
        Ns.append(Ns[-1] + dN)

    Ns[-1] = 0

    return ts, Ns

beta = 1
N0 = 25
dt = 0.001

mt = 0

for _ in range(10):
    ts, Ns = stoch_solve(dt,N0,beta)
    nts = np.linspace(0,ts[-1], 10000)
    if ts[-1] > mt: mt = ts[-1]
    plt.plot(nts, [Ns[np.argmax(ts>=T)] for T in nts], label = "Diskretna rešitev")

ex_ts = np.linspace(0,mt,100)
ex_Ns = exact_sol(ex_ts,beta,N0)
plt.plot(ex_ts, ex_Ns, label = "Zvezna rešitev", linewidth = 2, color = "black")

plt.xlabel("Čas")
plt.ylabel("Populacija")
plt.title(f"$\\beta = {beta}$, $N_0 = {N0}$, $\Delta t = {dt}$")

plt.ylim(bottom=0)

#plt.legend()

plt.savefig(f"b{beta}dt{dt}N0{N0}.png")
plt.show()