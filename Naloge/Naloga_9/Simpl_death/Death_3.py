import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re
import seaborn as sns


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
N0 = 250
dts = [0.01, 0.1, 0.5, 0.8, 1]
dts = [0.1]

palette = sns.color_palette(n_colors=len(dts))

for i in tqdm(range(len(dts))):
    dt = dts[i]
    endtimes = []

    for _ in tqdm(range(100000)):
        ts, Ns = stoch_solve(dt,N0,beta)
        endtimes.append(ts[-1])

    endtimes = np.array(endtimes)
    weights = np.ones_like(endtimes) / len(endtimes)
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), weights = weights, histtype="step", color=palette[i])
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), weights = weights, histtype="stepfilled", label=f"$\\Delta t = {dt}$", alpha = 0.4, color=palette[i])

#plt.legend()
plt.title(f"$\\beta = {beta}$, $N(0) = {N0}$")
plt.xlabel("Čas izumrtja")
plt.ylabel("Verjetnostna porazdelitev")

plt.show()

