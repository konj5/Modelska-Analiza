import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

def renorm(x):
    return x/np.sum(x)

br = 4
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
tmax = 4
x = np.zeros((Nmat, int(tmax/dt)))
x[N0,0]=1
for i in tqdm(range(1, len(x[0,:]))):
    #print(x[:,i-1])
    #print(np.sum(x[:,i-1]))
    x[:,i] = renorm(M.dot(x[:,i-1]))


cmap = cm.get_cmap("viridis")
norm = colors.Normalize(0,tmax)

fig, ax = plt.subplots(1,1)






for i in [k for k in np.int64(np.pow(np.linspace(1,(len(x[0,:])-1)**(1/2),10), 2))]:
    ax.plot(x[:,i], color = cmap(norm(tmax * i/len(x[0,:]))))


fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label = "Čas")
ax.set_xlabel("Komponenta $\\vec{x}$")
ax.set_ylabel("Verjetnost")
ax.set_xlim(-10,N0+10)

plt.show()