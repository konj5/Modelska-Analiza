import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

from scipy import fft

signal0 = []
with open("Naloge\\Naloga_10\\Wiener\\signal0.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        signal0.append(np.float64(line))

signal1 = []
with open("Naloge\\Naloga_10\\Wiener\\signal1.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        signal1.append(np.float64(line))

signal2 = []
with open("Naloge\\Naloga_10\\Wiener\\signal2.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        signal2.append(np.float64(line))

signal3 = []
with open("Naloge\\Naloga_10\\Wiener\\signal3.dat", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        signal3.append(np.float64(line))

tau = 16
def übertragungsfunktion(t):
    return 1/2/tau * np.exp(-np.abs(t)/tau)

def übertragungsfunktion_fourier(f):
    return 1/(1+4*f**2*np.pi**2*tau**2)


def dekonvolution0(c, n):

    r = np.array([übertragungsfunktion(t) for t in range(len(c))])

    for i in range(int(len(c)/2)):
        r[-i]=r[i]

    C = fft.fft(c)

    N = np.array([n(f) for f in range(len(C))])

    Phi = 1 - np.abs(N)**2/np.abs(C)**2

    return fft.ifft(C * Phi/fft.fft(r))


plt.plot(signal3, label = "signal3.dat")
plt.plot(signal2, label = "signal2.dat")
plt.plot(signal1, label = "signal1.dat")
plt.plot(signal0, label = "signal0.dat")
plt.legend()
plt.show()


plt.figure(figsize=(5,3.5))
plt.plot(dekonvolution0(signal0, lambda x:0), label = "Dekonvolirani signal")
plt.plot(signal0, label = "Merjeni signal")
plt.legend()
plt.title("signal0.dat")
plt.show()

plt.figure(figsize=(5,3.5))
plt.plot(dekonvolution0(signal1, lambda x:0), label = "Dekonvolirani signal")
plt.plot(signal1, label = "Merjeni signal")
plt.legend()
plt.title("signal1.dat")
plt.show()

plt.figure(figsize=(5,3.5))
plt.plot(dekonvolution0(signal2, lambda x:0), label = "Dekonvolirani signal")
plt.plot(signal2, label = "Merjeni signal")
plt.legend()
plt.title("signal2.dat")
plt.show()

plt.figure(figsize=(5,3.5))
plt.plot(dekonvolution0(signal3, lambda x:0), label = "Dekonvolirani signal")
plt.plot(signal3, label = "Merjeni signal")
plt.legend()
plt.title("signal3.dat")
plt.show()
