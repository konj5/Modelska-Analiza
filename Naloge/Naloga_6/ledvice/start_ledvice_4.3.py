import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

#BASIC ENORAZDELČNI MODEL

with open("Naloge\\Naloga_6\\ledvice.dat", mode = "r") as f:
    f.readline()
    lines = f.readlines()

#Extract data
ts, ys = [], []
for line in lines:
    line = line.strip()
    line = line.replace(" ", "\t")
    split = re.split("\t+", line)
    ts.append(float(split[0])); ys.append(float(split[1]))
ts = np.array(ts); ys = np.array(ys)


#Solve
from scipy.optimize import least_squares

def F(c):
    N0, L0, N1, L1, C0 = c

    return (N0 * np.exp(-L0*ts) + N1 * np.exp(-L1*ts) + C0  - ys)

def f(x):
    return N0 * np.exp(-L0*x) + N1 * np.exp(-L1*x) + C0 

L0s = np.linspace(0.1/10,0.1*10,200)
L1s = np.linspace(0.1/10,0.1*10,200)

N00 = 3000
L00 = 0.008
N10 = 8000
L10 = 0.000001
C00 = 100

xdata, ydata, zdata = [], [], []
for i in tqdm(range(len(L0s))):
    L00 = L0s[i]
    for j in tqdm(range(len(L1s)), leave=False):
        L10 = L1s[j]
        sol = least_squares(F, x0=[N00, L00, N10, L10, C00], method='lm')
        N0, L0, N1, L1, C0 = sol.x

        hi2 = np.sum((ys-f(ts))**2)
        xdata.append(L00)
        ydata.append(L10)
        zdata.append(hi2)

cmap = cm.get_cmap("viridis")
norm = colors.LogNorm(np.min(zdata), np.max(zdata))

plt.scatter(xdata,ydata, s=5, c=zdata, cmap=cmap, norm=norm)
plt.colorbar(label = "$\\chi^2$")
plt.xlabel("Začetni $L_0$")
plt.ylabel("Začetni $L_1$")
plt.show()