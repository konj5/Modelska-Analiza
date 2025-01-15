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

data_borza = []
with open("Naloge\\Naloga_12\\borza.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        data_borza.append(float(line))
data_borza = np.array(data_borza)



data_luna = []
lunatimes = []
with open("Naloge\\Naloga_12\\luna.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines[2:]:
        day, hour, dec = line.split(" ")

        data_luna.append(float(dec))

data_luna = np.array(data_luna)
lunatimes = np.array(lunatimes)


data_wolf = []

with open("Naloge\\Naloga_12\\Wolf_number.dat", mode = "r") as f:
    lines = f.readlines()

    for line in lines:
        hateyou = line.split(" ")

        data_wolf.append(float(hateyou[-1]))

data_wolf = np.array(data_wolf)


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
        poledata.append(np.array((xs,ys)))
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


def predict_RE(S, order, splitoff):
    ak, var = alg.AR_est_YW(S,order)

    poles = poles2(S,[order])[0]

  
    for i in range(len(poles[0,:])):
        pole = poles[:,i]
        print(pole)
        if np.linalg.norm(pole) > 1:

            """z = pole[0] + 1j * pole[1]
            z = z/np.conj(z)
            poles[:,i] = [np.real(z), np.imag(z)]"""
            poles[:,i] = pole / np.linalg.norm(pole)
           
        print(poles[:,i])

    zs = poles[0,:] + 1j * poles[1,:]

    new_coefs = np.poly(zs)[1:]
    
    ak = new_coefs[::-1]

    N = len(S)
    newS = np.zeros_like(S)
    newS[0:int(splitoff*N)] = S[0:int(splitoff*N)]

    ak = ak[::-1]
    for i in range(int(splitoff*N), N):
        newS[i] = np.sum(ak * newS[i-order:i])
    
    return newS, poles

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

sigma = 2

data = np.array([np.sin(x) for x in np.linspace(0,2*np.pi * 10,2000)])
data += (2*np.random.random(len(data))-1) * sigma
#data = data_co2[:,1]
#data = data2
#data = data3

#data = data_borza
#data = data_luna
#data = data_wolf

half = len(data)//2

result = np.polyfit([i for i in range(len(data))], data, deg=1)
lincoefs = result
lindata = [lincoefs[0] * i + lincoefs[1] for i in range(len(data))]

result = np.polyfit([i for i in range(len(data))], data, deg=2)
quadcoefs = result
quaddata = [quadcoefs[0] * i**2 + quadcoefs[1] * i + quadcoefs[2] for i in range(len(data))]



xs = [i for i in range(len(data))]

plt.plot(xs[half:], data[half:], c = "black", label = "signal")
#plt.plot(xs[half:], data[half:], c = "black", label = "signal")

#lindata = quaddata
lindata = np.zeros_like(data)
lindata = np.ones_like(data) * np.average(data)
data = data-lindata

ords = [100,20,10,3]
poles = poles2(data, ords)

newS = predict(data, 100, start)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 100")

newS = predict(data, 20, start)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 20")

newS = predict(data, 10, start)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 10")

newS = predict(data, 3, start)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 3")

plt.legend()

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

plt.figure(figsize=[5,5])

for i,poledata in enumerate(poles):
    plt.scatter(poledata[0], poledata[1], label = f"k ={ords[i]}")

x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1
plt.contour(X,Y,F,[0])
plt.legend()

plt.show()


#################
ords = [20,10,3]
plt.figure(figsize=[8,3])

plt.plot(xs[half:], (data+lindata)[half:], c = "black", label = "signal")
poles = []
"""newS, poledata = predict_RE(data, 100, start)
poles.append(poledata)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 100")"""

newS, poledata = predict_RE(data, 20, start)
poles.append(poledata)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 20")

newS, poledata = predict_RE(data, 10, start)
poles.append(poledata)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 10")

newS, poledata = predict_RE(data, 3, start)
poles.append(poledata)

plt.plot(xs[half:], newS[half:]+lindata[half:], label = "k = 3")

plt.legend()

plt.ylim(np.min(data+lindata), np.max(data+lindata))

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

plt.figure(figsize=[5,5])

for i,poledata in enumerate(poles):
    print(poledata.shape)
    plt.scatter(poledata[0,:], poledata[1,:], label = f"k ={ords[i]}")

x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1
plt.contour(X,Y,F,[0])
plt.legend()
plt.show()
        


