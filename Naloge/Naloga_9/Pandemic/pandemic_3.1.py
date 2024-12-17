import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)







elimtimes = np.load("elimtimes1.npy")
Dstarts = np.load("dstarts1.npy")
deltas = np.load("deltas1.npy")




cmap = cm.get_cmap("viridis")


norm = colors.LogNorm(np.min(elimtimes), np.max(elimtimes))

plt.imshow(elimtimes + 0.01, cmap=cmap, norm=norm, origin="lower", extent=[Dstarts[0], Dstarts[-1], deltas[0], deltas[-1]], aspect="auto")
plt.colorbar()
plt.xlabel("$D(0)$")
plt.ylabel("$\\delta$")
plt.title("Čas eliminacije bolezni")

plt.show()