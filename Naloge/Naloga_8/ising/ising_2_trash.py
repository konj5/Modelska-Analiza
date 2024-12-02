import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re

def H_old(s):
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

def H(s):
    E = 0
    
    E += -J*(np.sum(s[1:,:] * s[:-1,:]) + np.sum(s[:-1,:] * s[1:,:])   +   np.sum(s[:,1:] * s[:,:-1]) + np.sum(s[:,:-1] * s[:,1:]))
    E += -hz*np.sum(s)
    return E

def H_squared(s):
    Jsisj = -J*(np.sum(s[1:,:] * s[:-1,:]) + np.sum(s[:-1,:] * s[1:,:])   +   np.sum(s[:,1:] * s[:,:-1]) + np.sum(s[:,:-1] * s[:,1:]))
    hsi = -hz*np.sum(s)

    return Jsisj**2 + 2*Jsisj*hsi + hsi**2


def flip_1(s):
    E0 = H(s)
    n = 0

    while(True):
        i = np.random.randint(s.shape[0])
        j = np.random.randint(s.shape[1])

        š = np.copy(s)
        š[i,j] = 1 if s[i,j] == -1 else -1
        E1 = H(š)


        if E1-E0 < 0: break
        elif np.random.random() <= np.exp(-(E1-E0)/kbT): break
        
        if n == 102:
            break

        n += 1


    """fig, axs = plt.subplots(1,2)

    axs[0].imshow(s, cmap=cm.get_cmap("binary"))
    axs[0].set_xticks([]); axs[0].set_yticks([])

    axs[1].imshow(š, cmap=cm.get_cmap("binary"))
    axs[1].set_xticks([]); axs[1].set_yticks([])

    print(f"{E0:0.3f}   {E1:0.3f}")
    print(n)

    plt.show()"""
    
    return E1, š, n

def run_procedure(Nmin):
    s = np.random.choice([-1,1], size=shape)
    E0 = H(s)

    Elist = [E0]
    slist = [s]

    n = 0
    for _ in tqdm(range(Nmin+2), leave=False):
        E1, š, Ntries = flip_1(s)

        #if Ntries > 100: break

        #print(f"{E1:0.2f}  {np.abs(E0-E1):0.2f}   n={n:0.0f}")

        if n > Nmin: break

        """if len(Es) > 50:
            print(f"{E1:0.2f}  {np.abs(E0-E1):0.2f}   n={n:0.0f}   {np.abs(np.average(Es[-20:]) - np.average(Es[-5:])):0.4f}")
            if np.abs(np.average(Es[-50:]) - np.average(Es[-10:])) < 0.1:
                
                break"""
        
        #print(DEs[-6:])
        #print([0 for i in range(6)])

        if np.abs(E0-E1) < 0 and n > Nmin:
            break

        s = š
        E0 = E1
        Elist.append(E0)
        slist.append(s)

        n += 1

    return E1, š, Elist, slist

J = 1
hz = 0

kbT_C = 2.269186
shape = [10,10]

kbTs = np.linspace(0.01, 3, 10) * kbT_C

E_avg, M_avg = [],[]
X,c = [],[]

for i in tqdm(range(len(kbTs))):
    kbT = kbTs[i]
    Es = []
    Ms = []
    for j in tqdm(range(20), leave=False):
        E,s, Elist, slist = run_procedure(1000)

        N = shape[0]*shape[1]
        E = E/N
        M = np.sum(s)/N

        Es.append(E)
        Ms.append(M)

    E_avg.append(np.average(Es))
    M_avg.append(np.average(Ms))

    X.append((np.std(Ms)**2-np.average(Ms)**2)/(kbT)) 
    c.append((np.std(Es)**2-np.average(Es)**2)/(kbT**2))

    print("\n\n")
    print(E_avg[-1])
    print(M_avg[-1])
    print(X[-1])
    print(c[-1])
    print("\n\n")

plt.plot(kbTs,E_avg)
plt.show()

plt.plot(kbTs,M_avg)
plt.show()

plt.plot(kbTs,X)
plt.show()

plt.plot(kbTs,c)
plt.show()