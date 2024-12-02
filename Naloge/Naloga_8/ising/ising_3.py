import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re



def energija(s):
    E = 0
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for delta in [[1,0], [0,1], [-1,0], [0,-1]]:
                try:
                    E += -J * s[i,j] * s[i+delta[0], j+delta[1]]
                except IndexError:
                    pass

            E += -hz * s[i,j]

    return E

def energija_kvadrat(s):
    E = 0
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            E1 = 0
            for delta in [[1,0], [0,1], [-1,0], [0,-1]]:
                try:
                    E1 += -J * s[i,j] * s[i+delta[0], j+delta[1]]
                except IndexError:
                    pass

            E1 += -hz * s[i,j]
            E += E1**2
    return E

def magnetizacija(s):
    return np.sum(s)/N

def magnetizacija_kvadrat(s):
    return np.sum(s**2)/N

def kapaciteta(s):
    return (energija_kvadrat(s) - energija(s)**2)/(N * kbT**2)

def susceptibilnost(s):
    return (energija_kvadrat(s) - energija(s)**2)/kbT**2

def delta_energija(s,i,j):
    val = 0
    for delta in [[1,0], [0,1], [-1,0], [0,-1]]:
        try:
            val += s[i+delta[0], j+delta[1]]
        except IndexError:
            pass
    return 2*s[i,j] * (J * val + hz)

def run_procedure(Nmin):
    s = np.random.choice([-1,1], (n,n))
    ultradone = False

    for _ in tqdm(range(Nmin), leave=False, desc = "procedure"):
        if ultradone:
            break

        done = False
        k = 0
        while(not done):
            i,j = np.random.randint(0,n), np.random.randint(0,n)

            dE = delta_energija(s,i,j)

            if np.random.random() < np.exp(-dE/kbT):
                done = True
                s[i,j] *= -1
            k +=1
            if k > 1000:
                ultradone = True
                break

    return s


J = -1

hzs = np.linspace(0,2,2)

kbTs = np.linspace(0.1,5,10)

n = 100
N = n**2

eavg = np.zeros((len(hzs), len(kbTs)))
Mavg = np.zeros((len(hzs), len(kbTs)))
sus = np.zeros((len(hzs), len(kbTs)))
cap = np.zeros((len(hzs), len(kbTs)))

for i in tqdm(range(len(kbTs))):
    for j in tqdm(range(len(hzs)), leave=False):
        kbT = kbTs[i]
        hz = hzs[j]

        s = run_procedure(n**2 * 3)

        eavg[j,i] = energija(s)
        Mavg[j,i] = magnetizacija(s)
        sus[j,i] = susceptibilnost(s)
        cap[j,i] = kapaciteta(s)

for i in range(len(hzs)):
    plt.plot(kbTs, eavg[i,:], label = f"H = {hzs[i]}")
plt.legend()
plt.show()

for i in range(len(hzs)):
    plt.plot(kbTs, Mavg[i,:], label = f"H = {hzs[i]}")
plt.legend()
plt.show()

for i in range(len(hzs)):
    plt.plot(kbTs, sus[i,:], label = f"H = {hzs[i]}")
plt.legend()
plt.show()

for i in range(len(hzs)):
    plt.plot(kbTs, cap[i,:], label = f"H = {hzs[i]}")
plt.legend()
plt.show()


