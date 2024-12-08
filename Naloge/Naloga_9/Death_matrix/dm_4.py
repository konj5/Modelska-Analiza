import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

def renorm(x):
    return x/np.sum(x)

br = 0
bs = br +1
dt = 0.0001



N0 = 250
Nmat = int(1.5 * N0)
M = np.zeros((Nmat, Nmat))

print(1-br*N0*dt-bs*N0*dt)
#assert 1-br*N0*dt-bs*N0*dt > 0.9

Rn = np.array([0 if N == 0 else br*N*dt for N in range(0,Nmat)])
Sn = np.array([0 if N == 0 else bs*N*dt for N in range(0,Nmat)])

M += np.diag(1-Rn-Sn, 0) + np.diag(Rn[:-1], -1) + np.diag(Sn[1:], 1)

#print(M)
tmax = 12
x1 = np.zeros((Nmat, int(tmax/dt)))
x1[N0,0]=1
for i in tqdm(range(1, len(x1[0,:]))):
    #print(x[:,i-1])
    #print(np.sum(x[:,i-1]))
    x1[:,i] = renorm(M.dot(x1[:,i-1]))


br = 4
bs = br +1
dt = 0.00001



N0 = 250
Nmat = int(1.5 * N0)
M = np.zeros((Nmat, Nmat))

print(1-br*N0*dt-bs*N0*dt)
#assert 1-br*N0*dt-bs*N0*dt > 0.9

Rn = np.array([0 if N == 0 else br*N*dt for N in range(0,Nmat)])
Sn = np.array([0 if N == 0 else bs*N*dt for N in range(0,Nmat)])

M += np.diag(1-Rn-Sn, 0) + np.diag(Rn[:-1], -1) + np.diag(Sn[1:], 1)

#print(M)

x2 = np.zeros((Nmat, int(tmax/dt)))
x2[N0,0]=1
for i in tqdm(range(1, len(x2[0,:]))):
    #print(x[:,i-1])
    #print(np.sum(x[:,i-1]))
    x2[:,i] = renorm(M.dot(x2[:,i-1]))



plt.plot(np.linspace(0,tmax,len(x1[0,:])-1), (x1[0,1:]-x1[0,:-1])/dt/11/0.88*0.983, label = "$\\beta_r = 0$, $\\beta_s = 1$")


plt.plot(np.linspace(0,tmax,len(x2[0,:])-1), (x2[0,1:]-x2[0,:-1])/dt, label = "$\\beta_r = 4$, $\\beta_s = 5$")

##########################

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
    


##################

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

import seaborn as sns
palette = sns.color_palette(n_colors=len(dts)+1)

for i in tqdm(range(len(dts))):
    dt = dts[i]
    endtimes = []

    for _ in tqdm(range(100000)):
        ts, Ns = stoch_solve(dt,N0,beta)
        endtimes.append(ts[-1])

    endtimes = np.array(endtimes)
    weights = np.ones_like(endtimes) / len(endtimes)
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density=True, histtype="step", color=palette[i])
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density=True, histtype="stepfilled", alpha = 0.4, color=palette[i])


#############
##################

def exact_sol(t,beta,N0):
    return N0 * np.exp(-beta*t)

def stoch_solve(dt, N0, beta):
    ts = [0]
    Ns = [N0]

    while Ns[-1] > 0:


        ts.append(ts[-1] + dt)
        Ns.append(Ns[-1] + -np.random.poisson(5 * Ns[-1] * dt) +np.random.poisson(4 * Ns[-1] * dt))

    Ns[-1] = 0

    return ts, Ns


beta = 1
N0 = 250
dts = [0.01, 0.1, 0.5, 0.8, 1]
dts = [0.1]

import seaborn as sns
palette = sns.color_palette(n_colors=len(dts)+1)

for i in tqdm(range(len(dts))):
    dt = dts[i]
    endtimes = []

    for _ in tqdm(range(100000)):
        ts, Ns = stoch_solve(dt,N0,beta)
        endtimes.append(ts[-1])

    endtimes = np.array(endtimes)
    weights = np.ones_like(endtimes) / len(endtimes)
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density=True, histtype="step", color=palette[i+1])
    plt.hist(endtimes, np.linspace(0,np.max(endtimes),30), density=True, histtype="stepfilled", alpha = 0.4, color=palette[i+1])


#############



plt.legend()

plt.xlabel("Čas")
plt.ylabel("Porazdelitev časov izumrtja")


plt.show()

plt.plot(np.linspace(0,tmax,len(x1[0,:])), x1[0,:], label = "$\\beta_r = 0$, $\\beta_s = 1$")


plt.plot(np.linspace(0,tmax,len(x2[0,:])),  x2[0,:], label = "$\\beta_r = 4$, $\\beta_s = 5$")


plt.legend()

plt.xlabel("Čas")
plt.ylabel("Kumulativna porazdelitev časov izumrtja")


plt.show()

