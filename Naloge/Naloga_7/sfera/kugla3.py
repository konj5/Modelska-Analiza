import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

def get_random_trio(lp):
    #r,costheta,s
    return [np.random.random()**(1/3), 2*np.random.random()-1, -lp*np.log(np.random.random())]

def run_one_particle(lp):
    r, costheta, s = get_random_trio(lp)
    return int(s > np.sqrt(r**2*(costheta**2-1)+R**2)-r*costheta)

def run_for_N(N,lp):
    escapees = 0
    for _ in range(N):
        escapees += run_one_particle(lp)

    return escapees/N
    
R = 1
N = 10**4

ls = np.linspace(0,10,100)
lRs = np.linspace(0,10,100)

ratios = np.zeros((len(ls), len(lRs)))
xdata, ydata, zdata = [], [], []
for i in tqdm(range(len(ls))):
    for j in tqdm(range(len(lRs)), leave=False):
        R = ls[i]/lRs[i]
        xdata.append(ls[i])
        ydata.append(lRs[j])
        ratios[i,j] = run_for_N(N, ls[i])
        zdata.append(ratios[i,j])

cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(ratios), np.max(ratios))

plt.scatter(xdata, ydata, s=5, c=zdata, cmap=cmap, norm=norm)

plt.colorbar(label="Delež pobeglih fotonov")

plt.legend()

plt.xlabel("$l_p$")
plt.ylabel("$\\frac{l_p}{R}$")
plt.show()