import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

np.set_printoptions(linewidth=600)

#BASIC ENORAZDELČNI MODEL

with open("Naloge\\Naloga_6\\thtg-xfp-thfp.dat", mode = "r") as f:
    f.readline()
    lines = f.readlines()

#Extract data
ttg, xfp, tfp= [], [], []
for line in lines:
    line = line.strip()
    split = re.split(" +", line)
    ttg.append(float(split[0])); xfp.append(float(split[1])); tfp.append(float(split[2]))
ttg = np.array(ttg); xfp = np.array(xfp); tfp = np.array(tfp)
er_ttg = np.ones_like(ttg)*180/np.pi/2000; er_xfp = np.ones_like(xfp); er_tfp = np.ones_like(tfp)*180/np.pi/200


#Get polyform
pmax = 2
npars = []
lintimes, svdtmes = [], []

for pmax in tqdm(range(0,25)):
    parts = []
    for p in range(0,pmax+1,1):
        for i in range(p+1):
            parts.append([p-i, i])

    def f(x,t,c="Ligma balčs"):
        def C(i):
            if c is None: return 1
            else:
                return c[i]

        val = 0
        for k, part in enumerate(parts):
            i,j = part
            val += C(k) * x**i * t**j
        return val

    #Define matrix
    N = len(ttg); M = len(parts); H = np.zeros((N,M))
    for i, part in enumerate(parts):
        H[:,i] = 1 * xfp**part[0] * tfp**part[1]

    b = np.array(ttg).T

    def f(x,t,c="Ligma balčs"):
        def C(i):
            if c is None: return 1
            else:
                return c[i]

        val = 0
        for k, part in enumerate(parts):
            i,j = part
            val += C(k) * x**i * t**j
        return val

    #Solve system
    from scipy.linalg import svd
    import time

    stime = time.time()
    for _ in range(1000):
        cov = np.linalg.inv(H.T.dot(H))
        a = cov.dot(H.T.dot(b))
    lintime =  time.time() - stime
    lintimes.append(lintime/1000)

    stime = time.time()
    U, s, Vh = svd(H)

    a = np.zeros(M)
    for i in range(M):
        a += U[:,i].dot(b)/s[i]  * Vh[i,:]

    cov = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                cov[i,j] += Vh[k,i] * Vh[k,j] / s[k]**2
    svdtime =  time.time() - stime
    svdtmes.append(svdtime)
    npars.append(len(parts))

plt.plot(npars, lintimes, label = "Lin. minimizacija")
plt.plot(npars, svdtmes, label = "SVD minimizacija")
plt.legend()
plt.yscale("log")
plt.xlabel("Št. parametrov")
plt.ylabel("Čas računanja [s]")
plt.show()