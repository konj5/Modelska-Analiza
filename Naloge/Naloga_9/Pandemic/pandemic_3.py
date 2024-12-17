import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)


def dN(x):
    sign = np.sign(x)
    return np.random.poisson(x)


dt = 0.001
tmax = 130


args = []


Dstart = 1000
Bstart = 10
Istart = 0
Mstart = 0

alpha = 0.001
beta = 0.1
gamma = 0.1
delta = 0.1

"""alpha = 0.001
beta = 0.1
gamma = 0.01
delta = 0.05"""

ts = np.arange(0,tmax,dt)
Ds = np.zeros_like(ts, dtype=np.int32); Ds[0] = Dstart
Bs = np.zeros_like(ts, dtype=np.int32); Bs[0] = Bstart
Is = np.zeros_like(ts, dtype=np.int32); Is[0] = Istart
Ms = np.zeros_like(ts, dtype=np.int32); Ms[0] = Mstart



Dstarts = np.arange(10,10000, (10000-10)//30)
deltas = np.linspace(0.0001,0.15,120)

elimtimes = np.zeros((len(Dstarts), len(deltas)))

for k in tqdm(range(len(Dstarts))):
    Dstart = Dstarts[k]
    for h in tqdm(range(len(deltas)), leave=False):
        delta = deltas[h]

        end_times = []
        end_surv = []

        max_times = []
        max_sick = []
        for j in tqdm(range(6), leave=False):
            for i in range(1,len(ts)):

                """print(alpha*Bs[i-1]*Ds[i-1]*dt)
                print(alpha)
                print(Bs[i-1])
                print(Ds[i-1])
                print(dt)
                print("\n")"""

                A = dN(alpha*Bs[i-1]*Ds[i-1]*dt)
                B = dN(beta*Bs[i-1]*dt)
                C = dN(gamma*Bs[i-1]*dt)
                D = dN(delta*Is[i-1]*dt)


                dD = -A + D
                dB = A - B - C
                dI = B - D
                dM = C

                #print(dD + dB + dM + dI)

                Ds[i] = Ds[i-1] + dD if Ds[i-1] + dD > 0 else 0
                Bs[i] = Bs[i-1] + dB if Bs[i-1] + dB > 0 else 0
                Is[i] = Is[i-1] + dI if Is[i-1] + dI > 0 else 0
                Ms[i] = Ms[i-1] + dM if Ms[i-1] + dM > 0 else 0

                assert Ds[-1] >= 0
                assert Bs[-1] >= 0
                assert Is[-1] >= 0
                assert Ms[-1] >= 0

            #Eliminacija bolezni
            """print("\n")
            print("\n")
            print(Bs[-1])
            print("\n")
            print("\n")"""
            if Bs[-1] != 0: 
                end_times.append(ts[-1])
                end_surv.append(Bs[0] + Ds[0] + Is[0] + Ms[0] - Ms[-1])
            else:
                end_times.append(ts[np.argmax(Bs == 0)])
                end_surv.append(Bs[0] + Ds[0] + Is[0] + Ms[0] - Ms[-1])

            #Vrh bolezni
            max_sick.append(np.max(Bs))
            max_times.append(ts[np.argmax(Bs==max_sick[-1])])
            
            """print("\n")

            print(end_times[-1])
            print(end_surv[-1])

            print(max_times[-1])
            print(max_sick[-1])"""

            """plt.plot(ts, Ds, label = "Dovzetni")
            plt.plot(ts, Bs, label = "Bolni")
            plt.plot(ts, Is, label = "Imuni")
            plt.plot(ts, Ms, label = "Mrtvi")
            plt.show()"""

        """print("\n")
        print("\n")
        print(end_times)
        print("\n")
        print("\n")"""
        elimtimes[k,h] = np.average(end_times)

np.save("elimtimes.npy", elimtimes)
np.save("dstarts.npy", Dstarts)
np.save("deltas.npy", deltas)


cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(elimtimes), np.max(elimtimes))

plt.imshow(elimtimes, cmap=cmap, norm=norm, origin="lower", extent=[Dstarts[0], Dstarts[-1], deltas[0], deltas[-1]], aspect="auto")
plt.colorbar()
plt.xlabel("$D(0)$")
plt.ylabel("$\\delta$")
plt.title("Čas eliminacije bolezni")

plt.show()