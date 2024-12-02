import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re



def length(r):
    return np.sum(np.sqrt((r[0,1:]-r[0,0:-1])**2 + (r[1,1:]-r[1,0:-1])**2))

def is_legal(r):
    return True

def procedure(r0, Nmin):
    r0 = r0.T
    L0 = length(r0)
    for _ in range(Nmin):
        while(True):
            i = np.random.randint(0,len(r0[0,:])-1)
            r = np.copy(r0); r[:,i], r[:,i+1] = r0[:,i+1], r0[:,i]
            L = length(r)

            if np.random.random() < np.exp((L0-L)/T):
                r0 = r
                L0 = L
                break

    return r0, L0