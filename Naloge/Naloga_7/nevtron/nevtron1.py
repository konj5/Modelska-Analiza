import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


alpha = 1/2
beta = 1/2
d = 1

lp = alpha * d



def run_single():
    x = 0
    dir = 1

    i = 0
    while(True):
        s = -lp * np.log(np.random.random())
        x += dir * s
        i += 1
        if x < 0:
            return False, i
        if x > d:
            return True, i
        
def run_N(N):
    data = []
    bounces = []
    for i in range(N):
        T,n = run_single()
        data.append(T)
        bounces.append(n)
    
    return data, bounces


#### ERROR

Ns = np.int32(10**np.linspace(1,6,5))[::-1]

for i in range(len(Ns)):
    