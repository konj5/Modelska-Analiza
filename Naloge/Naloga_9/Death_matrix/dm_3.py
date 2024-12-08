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
tmax = 6
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

beta = 1
N0 = 250
dt = 0.01

mt = 0

x3 = np.zeros((10000, 40))
for _ in tqdm(range(40)):
    ts, Ns = stoch_solve(dt,N0,beta)
    nts = np.linspace(0,tmax, 10000)
    if ts[-1] > mt: mt = ts[-1]
    x3[:,_]=np.array([Ns[np.argmax(ts>=T)] for T in nts])

x3 = x3


##########################

##########################

def exact_sol(t,beta,N0):
    return N0 * np.exp(-beta*t)

def stoch_solve(dt, N0, beta):
    ts = [0]
    Ns = [N0]

    while Ns[-1] > 0:

        ts.append(ts[-1] + dt)
        Ns.append(Ns[-1] - np.random.poisson(5 * Ns[-1] * dt) + np.random.poisson(4 * Ns[-1] * dt))

    Ns[-1] = 0

    return ts, Ns

beta = 1
N0 = 250
dt = 0.01

mt = 0

x4 = np.zeros((10000, 40))
for _ in tqdm(range(40)):
    ts, Ns = stoch_solve(dt,N0,beta)
    nts = np.linspace(0,tmax, 10000)
    if ts[-1] > mt: mt = ts[-1]
    x4[:,_]=np.array([Ns[np.argmax(ts>=T)] for T in nts])

x4 = x4


##########################

n1 = np.zeros_like(x1)
for j in range(len(x1[0,:])):
    n1[:,j] = np.array([i for i in range(len(x1[:,0]))])

n2 = np.zeros_like(x2)
for j in range(len(x2[0,:])):
    n2[:,j] = np.array([i for i in range(len(x2[:,0]))])



plt.plot(np.linspace(0,tmax,len(x1[0,:])), np.sum(x1*n1, axis=0), label = "$Matrike: \\beta_r = 0$, $\\beta_s = 1$", linestyle = "dashed")

plt.plot(np.linspace(0,tmax,len(x3[:,0])), np.average(x3, axis=1), label = "$Direktno: \\beta_r = 0$, $\\beta_s = 1$")

plt.plot(np.linspace(0,tmax,len(x2[0,:])), np.sum(x2*n2, axis=0), label = "$Matrike: \\beta_r = 4$, $\\beta_s = 5$", linestyle = "dashed")

plt.plot(np.linspace(0,tmax,len(x4[:,0])), np.average(x4, axis=1), label = "$Direktno: \\beta_r = 0$, $\\beta_s = 1$")

plt.legend()

plt.xlabel("Čas")
plt.ylabel("Povprečna vrenost populacije")

plt.show()
