import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

from scipy import fft

def square(j, N):
    if j < N and j > 0: return 1
    return 0

square = np.vectorize(square)

def Welch(j, N):
    return 1 - ((j-N/2)/(N/2))**2

def Hann(j, N):
    return 1/2 * (1-np.cos(2*np.pi*j/N))

def Batrtlett(j, N):
    return 1 - np.abs((j-N/2)/(N/2))

def Gauss(j, N, sigma):
    return (np.exp(-(j-N/2)**2/(2*sigma**2)) - np.exp(-(N/2)**2/(2*sigma**2))) / (1 - np.exp(-(N/2)**2/(2*sigma**2)))

def Circle(j,N):
    return 2/N*np.sqrt((N/2)**2-(j-N/2)**2)

Circle = np.vectorize(Circle)

def power(fj, w):
    N = len(fj)-1
    fj = np.array(fj)
    wj = np.array([w(j,N) for j in range(N+1)])
    Fk = fft.fft(wj * fj)/(N+1)
    W = np.sum(wj**2)


    Pk = []
    for k in range(0, (N+1)//2):
        Pk.append(1/(2*W) * (np.abs(Fk[k])**2 + np.abs(Fk[N-k])**2))

    return Pk

data2 = []
with open("Naloge\\Naloga_10\\spec_anal\\val2.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        data2.append(np.float64(line))

data3 = []
with open("Naloge\\Naloga_10\\spec_anal\\val3.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        data3.append(np.float64(line))


data2 = data2[0:len(data2)//12]
data3 = data3[0:len(data3)//12]

windows = [square, Welch, Hann, Batrtlett, Circle]
names = ["kvadrat", "Welch", "Hann", "Batrtlett", "krog"]

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs

for i, window in enumerate(windows):
    ax1.plot(power(data2, window), label = names[i])
    ax2.plot(power(data2, window), label = names[i])


ax1.set_xlabel("k")
ax1.set_ylabel("Moč")
ax2.set_xlabel("k")
ax2.set_ylabel("Moč")
ax2.set_yscale("log")
ax2.legend()
plt.show()

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs

for i, window in enumerate(windows):
    ax1.plot(power(data3, window), label = names[i])
    ax2.plot(power(data3, window), label = names[i])


ax1.set_xlabel("k")
ax1.set_ylabel("Moč")
ax2.set_xlabel("k")
ax2.set_ylabel("Moč")
ax2.set_yscale("log")
ax1.legend(loc = "upper right")
plt.show()

