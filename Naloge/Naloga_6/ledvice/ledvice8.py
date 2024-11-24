import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

#BASIC DVORAZDELČNI MODEL + KONST (SQRT)

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

    return (N0 * np.exp(-L0*np.sqrt(ts)) + N1 * np.exp(-L1*np.sqrt(ts)) + C0  - ys)

sol = least_squares(F, x0=[10000,0.01, 10000, 0, 2000], method='lm')

N0, L0, N1, L1, C0 = sol.x

#plot

a = 1.4
plt.figure(figsize=(4/a,2*3/a))

ax1 = plt.subplot(2,1,1)

ax1.scatter(ts, ys, color='black')

def f(x):
    return N0 * np.exp(-L0*np.sqrt(x)) + N1 * np.exp(-L1*np.sqrt(x)) + C0  

new_xs = np.linspace(min(ts), max(ts), 1000)
new_ys = f(new_xs)

import seaborn as sns
palette = sns.color_palette(None, 8)

color = palette[7]

ax1.plot(new_xs, new_ys, color = color)

hi2 = np.sum((ys-f(ts))**2)

ax2 = plt.subplot(2,1,2)

ax2.bar(ts, (ys-f(ts))**2,width=40, color = color)

#plt.title(f"Analitično prilagajanje: $N_0 = {N0:0.2f}$, $L_0={L0}$ \n $N_1 = {N1:0.2f}$, $L_1={L1}$, $C_0 = {C0:0.2f}$ $\\chi^2 = {hi2:0.2f}$")
ax1.set_title("$N_0 \cdot e^{-L_0\sqrt{t}} + N_1 \cdot e^{-L_1\sqrt{t}} + C_0$")
ax1.set_xticks([]);ax1.set_yticks([]); plt.tight_layout()
print(f"$N_0 = {N0:0.2f}$, $L_0={L0}$  $\\chi^2 = {hi2:0.2f}$")

plt.savefig("ledvice8.png")

plt.show()


