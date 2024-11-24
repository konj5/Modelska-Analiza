import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

#BASIC DVORAZDELČNI MODEL + KONST

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

sol = least_squares(F, x0=[10000,0.01, 10000, 0, 2000], method='lm')

N0, L0, N1, L1, C0 = sol.x

J = sol.jac
M = np.linalg.inv(J.T.dot(J))

#plot

a = 1.4
plt.figure(figsize=(4/a,2*3/a))

ax1 = plt.subplot(2,1,1)

ax1.scatter(ts, ys, color='black')

def f(x):
    return N0 * np.exp(-L0*x) + N1 * np.exp(-L1*x) + C0  

new_xs = np.linspace(min(ts), max(ts), 1000)
new_ys = f(new_xs)

import seaborn as sns
palette = sns.color_palette(None, 8)

color = palette[3]

ax1.plot(new_xs, new_ys, color = color)

hi2 = np.sum((ys-f(ts))**2)

ax2 = plt.subplot(2,1,2)

ax2.bar(ts, (ys-f(ts))**2,width=40, color = color)

#plt.title(f"Analitično prilagajanje: $N_0 = {N0:0.2f}$, $L_0={L0}$ \n $N_1 = {N1:0.2f}$, $L_1={L1}$, $C_0 = {C0:0.2f}$ $\\chi^2 = {hi2:0.2f}$")

ax1.set_title("$N_0 \cdot e^{-L_0t} + N_1 \cdot e^{-L_1t} + C_0$")
ax1.set_xticks([]);ax1.set_yticks([]); plt.tight_layout()
print(f"$N_0 = {N0:0.2f}$, $L_0={L0}$, $L_1={L1}$, $N_1 = {N1:0.2f}$,$C0 = {C0:0.2f}$  $\\chi^2 = {hi2:0.2f}$")

plt.savefig("ledvice4.png")

plt.show()

p = 5




temp = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[0]):
        temp[i,j] = M[i,j] / np.sqrt(M[i,i]*M[j,j])
M = temp

cmap = cm.get_cmap("plasma")
norm = colors.Normalize(0, np.max(np.abs(M)))
plt.imshow(np.abs(M), cmap=cmap, norm=norm, extent=[0,p,0,p])
M = M[:,::-1]


for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        plt.text(i+0.1, j+0.5, f"{M[i,j]:0.3f}", color='black')

# draw gridlines
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

labels = ["$N_0$","$L_0$","$N_1$","$L_1$","C_0",""]

plt.xticks([i for i in range(0,p+1,1)], labels)
plt.yticks([i for i in range(0,p+1,1)], labels[::-1])


###RELATIVE ERRORS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!???????????

plt.colorbar(label = "$|\\rho|$")

plt.show()

