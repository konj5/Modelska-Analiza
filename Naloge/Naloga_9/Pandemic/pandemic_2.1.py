import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)




end_times = np.load("end_times.npy")
end_surv = np.load("end_surv.npy")

max_times = np.load("max_times.npy")
max_sick = np.load("max_sick.npy")

alpha, beta, gamma, delta, dt, tmax = np.load("metadata.npy")



plt.hist(max_times, np.linspace(0,tmax,300), density=True, label = "Vrh epidemije")
plt.hist(end_times, np.linspace(0,tmax,300), density=True, label = "Konec epidemije")

print(f"maxval = {np.average(max_sick):0.1f} $\pm$ {np.std(max_sick):0.1f}")
print(f"survval = {np.average(end_surv):0.1f} $\pm$ {np.std(end_surv):0.1f}")


plt.xlabel("Verjetnostna porazdelitev")
plt.ylabel("Populacija")
plt.title(f"$\\alpha = {alpha}, \\beta = {beta}, \\gamma = {gamma}, \\delta = {delta}, \\Delta t = {dt}$")
plt.legend()
plt.tight_layout()
#plt.savefig(f"test dt {dt}.png")
plt.show()
