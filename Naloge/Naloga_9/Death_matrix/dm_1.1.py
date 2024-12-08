import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

def renorm(x):
    return x/np.sum(x)

br = 0
bs = br +1
dt = 0.0001



N0 = 250
Nmat = int(1.5 * N0)
M = np.zeros((Nmat, Nmat))

print(1-br*N0*dt-bs*N0*dt)
#assert 1-br*N0*dt-bs*N0*dt > 0.9

Rn = np.array([0 if N == 0 else br*N*dt for N in range(0,Nmat)])
Sn = np.array([0 if N == 0 else bs*N*dt for N in range(0,Nmat)])

M += np.diag(1-Rn-Sn, 0) + np.diag(Rn[:-1], -1) + np.diag(Sn[1:], 1)

#print(M)
tmax = 5.4
x = np.zeros((Nmat, int(tmax/dt)))
x[N0,0]=1
for i in tqdm(range(1, len(x[0,:]))):
    #print(x[:,i-1])
    #print(np.sum(x[:,i-1]))
    x[:,i] = renorm(M.dot(x[:,i-1]))

cmap = cm.get_cmap("hot")
norm = colors.LogNorm(10**-6,1); x += 10**-6
plt.imshow(x, aspect=120, norm=norm, cmap=cmap, origin="lower")
plt.xticks(np.int32(np.linspace(0,int(tmax/dt),10)), [f"{x:0.1f}" for x in np.linspace(0,int(tmax),10)])
plt.colorbar()

plt.title(f"$\\beta_s = {bs}$ $\\beta_r = {br}$")
plt.xlabel("Čas")
plt.ylabel("Komponente $\\vec{x}$")
plt.show()
