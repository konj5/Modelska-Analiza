import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re



def energija(s):
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

def delta_energija(s,i,j):
    val = 0
    for delta in [[1,0], [0,1], [-1,0], [0,-1]]:
        try:
            val += s[i+delta[0], j+delta[1]]
        except IndexError:
            pass
    return 2*s[i,j] * (J * val + hz)

def run_procedure(Nmin):
    s = np.random.choice([-1,1], (n,n))
    ultradone = False

    for _ in tqdm(range(Nmin), leave=False, desc = "procedure"):
        if ultradone:
            break

        done = False
        k = 0
        while(not done):
            i,j = np.random.randint(0,n), np.random.randint(0,n)

            dE = delta_energija(s,i,j)

            if np.random.random() < np.exp(-dE/kbT):
                done = True
                s[i,j] *= -1
            k +=1
            if k > 1000:
                ultradone = True
                break

    return s


J = -1

hz = 0

kbT = 0.1

n = 100

s = run_procedure(n**2 * 5)

plt.figure(figsize=(3,3))
plt.imshow(s, cmap=cm.get_cmap("binary"))
plt.title(f"$J={J}$, $H = {hz}$, $k_bT = {kbT}$")
plt.xticks([]); plt.yticks([])
plt.savefig(f"J{J}H{hz}kbT{kbT}.png")
plt.show()
