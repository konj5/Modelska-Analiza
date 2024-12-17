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

def power(fj):
    N = len(fj)-1
    fj = np.array(fj)
    Fk = fft.fft(fj)/(N+1)


    Pk = []
    for k in range(0, (N+1)//2):
        Pk.append(1/2 * (np.abs(Fk[k])**2 + np.abs(Fk[N-k])**2))

    return Pk

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

    #N = np.array([n(f) for f in range(len(C))])
    #N = np.sqrt(np.average(np.abs(C[100:-100])**2))




    N = np.sqrt(np.average(np.abs(power(c)[100:])))
    print(N)


    #Phi = 1 - np.abs(N)**2/np.abs(C)**2

    S = [10**(np.log10(0.23)+(x/65 * (np.log10(2.57e-12) - np.log10(0.23)))) for x in range(len(C))]
    Phi = S/(S + N**2)

    """plt.plot(power(c))
    plt.axhline(N**2,0,255)
    plt.yscale("log")
    plt.show()"""

    #x = 15
    #Phi = np.array([np.average(Phi[max(0,k-x):min(k+x,len(Phi))]) for k in range(len(Phi))])

    #Phi[64:-64] = 0
    #Phi[33:-33] = 0

    """plt.plot(S)
    plt.plot(Phi)
    plt.axhline(N**2,0,255)
    plt.yscale("log")
    plt.show()"""

    ax1.plot(power(c), label = "$|C|^2$")
    ax1.plot(Phi, label = "$\\Phi$")
    ax1.plot(S, label = "$|S|^2$")
    ax1.axhline(N**2,0,255, label = "$|N|^2$", c="black")
    ax1.set_yscale("log")
    ax1.set_xlabel("frekvenca")
    ax1.legend()
    ax1.set_ylim(np.min(power(c))/10)
    ax1.set_xlim(0,255)

    return fft.ifft(C * Phi/fft.fft(r))


"""plt.plot(signal3, label = "signal3.dat")
plt.plot(signal2, label = "signal2.dat")
plt.plot(signal1, label = "signal1.dat")
plt.plot(signal0, label = "signal0.dat")
plt.legend()
plt.show()"""


"""plt.figure(figsize=(5,3.5))
plt.plot(dekonvolution0(signal0, lambda x:0), label = "Dekonvolirani signal")
plt.plot(signal0, label = "Merjeni signal")
plt.legend()
plt.title("signal0.dat")
plt.show()"""

fig, axs = plt.subplots(1,2, figsize = (10,3)); ax1, ax2 = axs
ax2.plot(signal0, label = "Merjeni signal")
ax2.plot(dekonvolution0(signal0, lambda x:0), label = "Dekonvolirani signal")
ax2.set_xlabel("čas")
ax2.legend()
fig.suptitle("signal0.dat")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2, figsize = (10,3)); ax1, ax2 = axs
ax2.plot(signal1, label = "Merjeni signal")
ax2.plot(dekonvolution0(signal1, lambda x:0), label = "Dekonvolirani signal")
ax2.set_xlabel("čas")
ax2.legend()
fig.suptitle("signal1.dat")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2, figsize = (10,3)); ax1, ax2 = axs
ax2.plot(signal2, label = "Merjeni signal")
ax2.plot(dekonvolution0(signal2, lambda x:0), label = "Dekonvolirani signal")
ax2.set_xlabel("čas")
ax2.legend()
fig.suptitle("signal2.dat")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2, figsize = (10,3)); ax1, ax2 = axs
ax2.plot(signal3, label = "Merjeni signal")
ax2.plot(dekonvolution0(signal3, lambda x:0), label = "Dekonvolirani signal")
ax2.set_xlabel("čas")
ax2.legend()
fig.suptitle("signal3.dat")
plt.tight_layout()
plt.show()
