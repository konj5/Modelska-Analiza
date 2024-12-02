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

    Es = [E0]

    DEs = []
    terminate = False

    n = 0
    while(True):
        E1, š, Ntries = flip_1(s)

        #if Ntries > 100: break

        Es.append(E1)
        DEs.append(np.abs(Es[-1]-Es[-2]))

        print(f"{E1:0.2f}  {np.abs(E0-E1):0.2f}   n={n:0.0f}")

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
        n += 1

    return E1, š

J = 1
hz = 0

kbT = 0.1* 2.269186*J


shape = [100,100]

E,s = run_procedure(30000)


plt.imshow(s, cmap=cm.get_cmap("binary"))
plt.xticks([]); plt.yticks([])
plt.show()

