import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

import spectrum
from nitime import algorithms as alg

from scipy import signal as sig

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

"""x = 4.95
f1 = 20-x
f2 = 10+x
orders = [1,2,3,4,20]
orders = [10,20,30]
orders = [50,80,90]

plt.figure(figsize=(5,5))

xs = np.linspace(0,10,100)
peak1 = np.sin(f1 * xs)
peak2 = np.sin(f2 * xs)

signal = peak1 + peak2

for order in orders:
    fs, ps = spectre(signal, order)
    plt.plot(fs, ps, label = f"k = {order}")

plt.xticks([])
plt.yticks([])
plt.title(f"$\\Delta \\nu = {f1-f2:0.1f}$")
plt.legend()
plt.show()"""

plt.figure(figsize=(5,5))
xss =  np.linspace(0,5,100, endpoint=False)[::-1]
orderss = np.zeros_like(xss)
realvals = []
for i in tqdm(range(len(xss))):
    x = xss[i]
    f1 = 20-x
    f2 = 10+x
    
    orders = np.arange(1,100,1)
    xs = np.linspace(0,10,100)
    peak1 = np.sin(f1 * xs)
    peak2 = np.sin(f2 * xs)

    signal = peak1 + peak2


    endorder = -100
    for order in orders:
        fs, ps = spectre(signal, order)
        peaks, _ = sig.find_peaks(ps)

        """if len(peaks) != 3:
            print(f"no three {peaks}")
            plt.plot(ps)
            plt.scatter(peaks, ps[peaks])
            plt.show()
            continue

        p1 = peaks[0]; p2 = peaks[2]; middle = peaks[1]
        print("three")"""

        if len(peaks) != 2:
            continue

        if ps[peaks[0]] - ps[np.sum(peaks)//2] > 0.2 * ps[peaks[0]] and ps[peaks[1]] - ps[np.sum(peaks)//2] > 0.2 * ps[peaks[1]]:
            #print(f"{ps[peaks[0]]}  {ps[peaks[1]]}   {ps[np.sum(peaks)//2]}")
            print(f"{f1},  {f2}")
            print(endorder)
            #plt.plot(ps)
            #plt.scatter(peaks, ps[peaks])
            #plt.show()
            endorder = order
            print("acceptable")
            break

    orderss[i] = endorder
    if endorder != -100:
        realvals.append(i)
        




plt.scatter(10-2*xss[realvals], orderss[realvals], s = 12, color = "black")
plt.xlabel("$\\Delta \\nu$")
plt.ylabel("$k_{\\text{min}}$")
plt.gca().invert_xaxis()
plt.show()



