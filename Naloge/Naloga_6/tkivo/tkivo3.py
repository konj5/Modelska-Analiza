import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

# NUMERIČEN FIT ZAKOMPLICIRAN MODEL

with open("Naloge\\Naloga_6\\farmakoloski.dat", mode = "r") as f:
    f.readline()
    lines = f.readlines()

#Extract data
xs, ys, dys = [], [], []
for line in lines:
    line = line.strip()
    split = re.split("\t+", line)
    xs.append(float(split[0])); ys.append(float(split[1])); dys.append(float(split[2]))
xs = np.array(xs); ys = np.array(ys); dys = np.array(dys)

#Solve
from scipy.optimize import least_squares

def F(c):
    y0, a, p = c

    return (y0*xs**p/(xs**p + a**p) - ys)/dys

sol = least_squares(F, x0=[1,1,1], method='lm')

y0,a,p = sol.x

J = sol.jac
M = np.linalg.inv(J.T.dot(J))

#plot

plt.errorbar(xs, ys, dys, color='red', ls='', marker='o', capsize=5, capthick=1, ecolor='black')

def f(x):
    return y0*x**p/(x**p + a**p) 

new_xs = np.linspace(min(xs), max(xs), 1000)
new_ys = f(new_xs)
plt.plot(new_xs, new_ys)

hi2 = np.sum((ys-f(xs))**2 / dys**2)

plt.title(f"Numerično prilagajanje: $y_0 = {y0:0.0f}\\pm{np.sqrt(M[0,0]):0.1f}$, $a={a:0.1f}\\pm{np.sqrt(M[1,1]):0.2f}$, $p={p:0.2f}\\pm{np.sqrt(M[2,2]):0.2f}$ $\\chi^2 = {hi2:0.2f}$")

plt.xlabel("x")
plt.ylabel("y")


plt.show()


