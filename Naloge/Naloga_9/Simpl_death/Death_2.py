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
dt = 0.1

endtimes = []

for _ in tqdm(range(100000)):
    ts, Ns = stoch_solve(dt,N0,beta)
    endtimes.append(ts[-1])

endtimes = np.array(endtimes)
plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density = True, histtype="step", label="N(0) = 25")

beta = 1
N0 = 250
dt = 0.1


endtimes = []

for _ in tqdm(range(100000)):
    ts, Ns = stoch_solve(dt,N0,beta)
    endtimes.append(ts[-1])

endtimes = np.array(endtimes)
plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density = True, histtype="step", label="N(0) = 250")

beta = 1
N0 = 2500
dt = 0.1


endtimes = []

for _ in tqdm(range(100000)):
    ts, Ns = stoch_solve(dt,N0,beta)
    endtimes.append(ts[-1])

endtimes = np.array(endtimes)
plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density = True, histtype="step", label="N(0) = 2500")

plt.legend()
plt.title(f"$\\beta = {beta}$, $\Delta t = {dt}$")
plt.xlabel("Čas izumrtja")
plt.ylabel("Verjetnostna porazdelitev")

plt.show()

