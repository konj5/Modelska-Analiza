import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

import spectrum
from nitime import algorithms as alg

data2, data3, data_co2 = [], [], []

with open("Naloge\\Naloga_12\\val2.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data2.append(float(line))

with open("Naloge\\Naloga_12\\val3.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data3.append(float(line))


#*******************************************************************
#*** Atmospheric CO2 concentrations (ppmv) derived from in situ  ***
#*** air samples collected at Mauna Loa Observatory, Hawaii      ***
#***                                                             ***
#*** Source: C.D. Keeling                                        ***
#***         T.P. Whorf, and the Carbon Dioxide Research Group   ***
#***         Scripps Institution of Oceanography (SIO)           ***
#***         University of California                            ***
#***         La Jolla, California USA 92093-0444                 ***
#***                                                             ***
#*** May 2005                                                    ***
#***                                                             ***
#*******************************************************************
#Monthly values are expressed in parts per million (ppm) and reported in the 2003A SIO manometric mole 
#fraction scale.  The monthly values have been adjusted to the 15th of each month.  Missing values are 
#denoted by -99.99. 

with open("Naloge\\Naloga_12\\co2.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data_co2.append([float(line.split(" ")[0]), float(line.split(" ")[1]) if float(line.split(" ")[1]) != -99.99 else data_co2[-1][1]])

data2 = np.array(data2)
data3 = np.array(data3)     
data_co2 = np.array(data_co2)


def getR(S):
    N = len(S)
    def R(k):
        return 1/(N-k) * np.sum(S[k:] * S[:len(S)-k])
    
    return R
    
def getG(S, ak):
    R = getR(S)

    G = np.sqrt(R(0) + np.sum(ak * [R(k) for k in range(len(ak))]))

    return G


def getP(S, order):
    coefs, var, refcoefs = spectrum.aryule(S, order)

    G = getG(S, coefs)

    def P(w):
        return G**2 * np.abs(1 + np.sum(coefs * [np.exp(-1j * w * k) for k in range(len(coefs))]))**2
    return P

def spectre(S,order):
    #P = getP(S,order)
    #freqs = np.array([k/len(S) for k in range(len(S))])
    #Ps = np.array([P(freq) for freq in freqs])
    a,b = alg.AR_est_YW(S,order)
    f = alg.autoregressive.AR_psd(a,b)

    return f[0]/(2*np.pi), f[1]/np.sum(f[1])

def spectreFFT(S):
    fft_koncentracija=abs(np.fft.fft(S))**2
    fft_koncentracija=fft_koncentracija/sum(fft_koncentracija)
    f_cas=np.arange(len(fft_koncentracija))/len(fft_koncentracija)

    return f_cas,fft_koncentracija

def poles1(S, ords):
    poledata = []
    for i,order in enumerate(ords):
        coefs, var, refcoefs = spectrum.aryule(S, order)

        coefs_mod = np.ones(len(coefs)+1)
        coefs_mod[1:] = coefs

        zs = np.roots(coefs_mod)
        print(zs)

        xs = np.real(zs); ys = np.imag(zs)

        poledata.append((xs,ys))
    return poledata

def poles2(S, ords):
    poledata = []
    for i,order in enumerate(ords):
        coefs, var = alg.AR_est_YW(S,order)

        coefs_mod = np.ones(len(coefs)+1)
        coefs_mod[1:] = coefs

        zs = np.roots(coefs_mod)

        xs = np.real(zs); ys = np.imag(zs)
        poledata.append((xs,ys))
    return poledata
        


def predict(S, order, splitoff):
    ak, var = alg.AR_est_YW(S,order)


    N = len(S)
    newS = np.zeros_like(S)
    newS[0:int(splitoff*N)] = S[0:int(splitoff*N)]

    ak = ak[::-1]
    for i in range(int(splitoff*N), N):
        newS[i] = np.sum(ak * newS[i-order:i])
    
    return newS

def predict1(S, order, splitoff):
    ak, var, refcoefs = spectrum.aryule(S, order)


    N = len(S)
    newS = np.zeros_like(S)
    newS[0:int(splitoff*N)] = S[0:int(splitoff*N)]

    ak = ak[::-1]
    for i in range(int(splitoff*N), N):
        newS[i] = np.sum(ak * newS[i-order:i])
    
    return newS


"""fr, da = spectre(data2,100)
fig, axs = plt.subplots(1,2)
axs[0].plot(fr, da)
axs[1].plot(fr, da)
axs[1].set_yscale("log")

fr, da = spectreFFT(data2)
axs[0].plot(fr, da)
axs[1].plot(fr, da)
axs[1].set_yscale("log")

axs[0].set_xlim(xmax=0.5)
axs[1].set_xlim(xmax=0.5)

plt.show()"""

"""ords = [1,4,10,20]
poles = poles1(data2, ords)
fig, axs = plt.subplots(1,2)
for i,poledata in enumerate(poles):
    axs[0].scatter(poledata[0], poledata[1], label = f"{ords[i]}")

x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1
axs[0].contour(X,Y,F,[0])

poles = poles2(data2, ords)
for i,poledata in enumerate(poles):
    axs[1].scatter(poledata[0], poledata[1], label = f"{ords[i]}")

x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1
axs[1].contour(X,Y,F,[0])

axs[1].legend()

plt.show()"""

start = 0.5

plt.figure(figsize=[8,3])

#data = np.array([np.sin(x) for x in np.linspace(0,2*np.pi * 10,4000)])
data = data_co2[:,1]
#data = data2
#data = data3

half = len(data)//2


xs = [i for i in range(len(data))]

plt.plot(xs, data, c = "black", label = "signal")
#plt.plot(xs[half:], data[half:], c = "black", label = "signal")

newS = predict(data, 100, start)
plt.plot(xs[half:], newS[half:], label = "k = 100")

newS = predict(data, 20, start)
plt.plot(xs[half:], newS[half:], label = "k = 20")

newS = predict(data, 10, start)
plt.plot(xs[half:], newS[half:], label = "k = 10")

newS = predict(data, 3, start)
plt.plot(xs[half:], newS[half:], label = "k = 3")

plt.legend()

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
        


