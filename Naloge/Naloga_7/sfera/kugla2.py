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
N = 10**5

ls = np.linspace(0,6,40)
intls = np.linspace(0,6,100)

ratios = np.zeros(len(ls))
intrat = np.zeros(len(intls))
from scipy.integrate import dblquad
for i in tqdm(range(len(ls))):
    ratios[i] = run_for_N(N, ls[i])

for i in tqdm(range(len(intls))):
    
    intrat[i] = dblquad(
        func = lambda u,v: u**2 * np.exp((-np.sqrt(u**2*(v**2-1)+R**2)+u*v)/intls[i]),
        a=0,
        b=R,
        gfun=-1,
        hfun=1
    )[0] * 3/(2*R**3)
print(intls)
print(intrat)

plt.plot(intls, intrat, label = "Integral", color = "red")
plt.scatter(ls, ratios, label = "Monte-Carlo", s = 9)


plt.legend()

plt.xlabel("$l_p$")
plt.ylabel("Delež pobeglih fotonov")
plt.show()