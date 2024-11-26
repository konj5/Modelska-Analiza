import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


def run_single():
    x = 0
    dir = 1

    i = 0
    while(True):
        s = -lp * np.log(np.random.random())
        x += dir * s
        if x < 0:
            return False, i
        if x > d:
            return True, i
        
        i += 1
        dir = 1 if np.random.random() >= 0.5 else -1
        
def run_N(N):
    data = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T,n = run_single()
        data.append(int(T))
        bounces.append(n)
    
    return data, bounces


alpha = 1/2
beta = 1/2
d = 1

lp = alpha * d

N = 10**4
data, bounces = run_N(N)

print(f"T = {np.sum(data)/N}, R = {1-np.sum(data)/N}")


Nbs = np.arange(0,np.max(bounces)+1,1)
Bs, BsT, BsR = np.zeros_like(Nbs), np.zeros_like(Nbs), np.zeros_like(Nbs)
###Put into bins
bounces = np.array(bounces)
data = np.array(data)
for i in tqdm(range(len(Nbs))):
    Bs[i] = np.sum(np.int32(bounces == i))
    BsT[i] = np.sum(np.int32(bounces == i) * np.int32(data == 1))
    BsR[i] = Bs[i] - BsT[i] 
    


plt.bar(Nbs-0.3, Bs, label = "Skupaj", width = 0.3)
plt.bar(Nbs, BsT, label = "Prepuščen", width = 0.3)
plt.bar(Nbs+0.3, BsR, label = "Odbit", width = 0.3)
plt.xticks(Nbs)
plt.legend()

plt.xlabel("Število odbojev")
plt.ylabel("Število nevtronov")

plt.title(f"$N = 10^8$, $T = {np.sum(data)/N:0.3f}$, , $R = {1-np.sum(data)/N:0.3f}$")

#plt.yscale("log")
plt.show()

plt.bar(Nbs-0.3, Bs, label = "Skupaj", width = 0.3)
plt.bar(Nbs, BsT, label = "Prepuščen", width = 0.3)
plt.bar(Nbs+0.3, BsR, label = "Odbit", width = 0.3)
plt.xticks(Nbs)
plt.legend()

plt.xlabel("Število odbojev")
plt.ylabel("Število nevtronov")

plt.title(f"$N = 10^8$, $T = {np.sum(data)/N:0.3f}$, , $R = {1-np.sum(data)/N:0.3f}$")

plt.yscale("log")
plt.show()
