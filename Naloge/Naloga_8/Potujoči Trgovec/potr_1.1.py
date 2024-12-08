import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re



def length(r):
    #print(r)
    #print(r[0,:])
    return np.sum(np.sqrt((r[0,1:]-r[0,0:-1])**2 + (r[1,1:]-r[1,0:-1])**2))

def all_is_legal(r):
    return True

def procedure(r0, Nmin):
    r0 = r0.T
    L0 = length(r0)
    
    Lmin = L0
    rmin = r0
    for _ in tqdm(range(Nmin)):
        if _ in (np.int32(np.linspace(0,Nmin,5))):
                plt.figure(figsize=(4,4))
                plt.grid()
                for i in range(len(rmin[0,:])-1):
                    print((rmin[0,i], rmin[1,i]))
                    plt.arrow(rmin[0,i], rmin[1,i], rmin[0,i+1]-rmin[0,i], rmin[1,i+1]-rmin[1,i], head_width = 0.03, head_length=0.02, length_includes_head = True)
                plt.scatter(rmin[0,:],rmin[1,:])
                plt.title(f"korak {_}, dolžina = {Lmin:0.2f}")
                

                print("\n")
                print(rmin[:,:])
                print("\n")

                plt.savefig(f"1.1 step={_}.png")
                
                plt.show()

        while(True):
            i = np.random.randint(0,len(r0[0,:]))
            j = np.random.randint(0,len(r0[0,:]))
            while i == j:
                j = np.random.randint(0,len(r0[0,:]))

            r = np.copy(r0); r[:,i], r[:,j] = r0[:,j], r0[:,i]
            L = length(r)


            

            if L < Lmin:
                Lmin = L
                rmin = r

            if np.random.random() < np.exp((L0-L)/T):
                r0 = r
                L0 = L
                break

    return rmin, Lmin

T = 0.1
np.random.seed(1)
r0 = np.random.random((4,2))

Nmin = 50000
r, L = procedure(r0, Nmin)

plt.figure(figsize=(4,4))
plt.grid()

for i in range(len(r[0,:])-1):
    plt.arrow(r[0,i], r[1,i], r[0,i+1]-r[0,i], r[1,i+1]-r[1,i], head_width = 0.03, head_length=0.02, length_includes_head = True)

plt.scatter(r0.T[0,:],r0.T[1,:])
plt.title(f"korak {Nmin}, dolžina = {L:0.2f}")

plt.savefig(f"1.1 step={Nmin}.png")

plt.show()