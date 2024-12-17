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


windows = [square, Welch, Hann, Batrtlett, Circle]
names = ["kvadrat", "Welch", "Hann", "Batrtlett", "krog"]

N = 100
js = np.arange(0,N+1,1)

for i, wind in enumerate(windows):
    plt.plot(js, wind(js,N), label = names[i])

plt.legend()
plt.show()

for sigma in N * np.array([0.5,0.2,0.1,0.01,0.001]):
    plt.plot(js, Gauss(js,N,sigma), label = f"{sigma/N}$\\cdot N$")

plt.legend()
plt.show()