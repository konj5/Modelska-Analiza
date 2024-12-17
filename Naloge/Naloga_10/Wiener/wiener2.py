import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import seaborn as sns
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

def power(fj):
    N = len(fj)-1
    fj = np.array(fj)
    Fk = fft.fft(fj)/(N+1)


    Pk = []
    for k in range(0, (N+1)//2):
        Pk.append(1/2 * (np.abs(Fk[k])**2 + np.abs(Fk[N-k])**2))

    return Pk


plt.plot(power(signal3), label = "signal3.dat")
plt.plot(power(signal2), label = "signal2.dat")
plt.plot(power(signal1), label = "signal1.dat")
plt.plot(power(signal0), label = "signal0.dat")
plt.legend(loc="upper right")
plt.xlabel("Frekvenca")
plt.ylabel("$|C(f)|^2$")
plt.yscale("log")
plt.show()


pallete = sns.color_palette(n_colors=5)

print(pallete)

xs = np.linspace(0, 65, 100)
log10_data = np.log10(0.23)+(np.linspace(0, 65, 100)/65 * (np.log10(2.57e-12) - np.log10(0.23)))

plt.plot(xs, 10**log10_data, linestyle = "dashed", label = "$|S(f)|^2$", c = pallete[0])
plt.axhline(6.04e-9, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal0.dat", c = pallete[1])
plt.axhline(1.32e-7, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal1.dat", c = pallete[2])
plt.axhline(6.04e-6, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal2.dat", c = pallete[3])
plt.axhline(0.0014, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal3.dat", c = pallete[4])
plt.xlim(0,255)
plt.legend()
plt.xlabel("Frekvenca")
plt.ylabel("$|C(f)|^2$")
plt.yscale("log")
plt.show()

plt.plot(xs, 10**log10_data, linestyle = "dashed", label = "$|S(f)|^2$", c = pallete[0])
plt.axhline(6.04e-9, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal0.dat", c = pallete[1])
plt.axhline(1.32e-7, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal1.dat", c = pallete[2])
plt.axhline(6.04e-6, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal2.dat", c = pallete[3])
plt.axhline(0.0014, 0, 255, linestyle = "dashed", label = "$|N(f)|^2$ - signal3.dat", c = pallete[4])
plt.xlim(0,255)
plt.plot(power(signal3), label = "signal3.dat")
plt.plot(power(signal2), label = "signal2.dat")
plt.plot(power(signal1), label = "signal1.dat")
plt.plot(power(signal0), label = "signal0.dat")
plt.legend(loc="upper right")
plt.xlabel("Frekvenca")
plt.ylabel("$|C(f)|^2$")
plt.yscale("log")
plt.show()