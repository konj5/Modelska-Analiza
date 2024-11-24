import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

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
N = len(ttg); M = len(parts); A = np.zeros((N,M))
for i, part in enumerate(parts):
    A[:,i] = 1 * xfp**part[0] * tfp**part[1]

b = np.array(ttg).T

#Solve
from scipy.linalg import svd

U, s, Vh = svd(A)

a = np.zeros(M)
for i in range(M):
    a += U[:,i].dot(b)/s[i]  * Vh[i,:]

cov = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        for k in range(M):
            cov[i,j] += Vh[k,i] * Vh[k,j] / s[i]**2

ax = plt.axes(projection='3d')

ax.scatter(xfp,tfp,ttg)

x = np.linspace(min(xfp), max(xfp), 100)
y = np.linspace(min(tfp), max(tfp), 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y, c = a)

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('$x_{fp}$')
ax.set_ylabel('$\\theta_{fp}$')
ax.set_zlabel('$\\theta_{tg}$')


hi2 = np.sum((ttg-f(xfp, tfp, c=a))**2)
hi2 = (A.dot(a)-b).dot(A.dot(a)-b)
print(hi2)

plt.show()

## plot chi^2 by q

##plot coefitients for q = 5 in grid with color gradient. Remove the ones that dont contribute!

### Reši z kalmanom (problem je linearen!)





data = np.ones((p+1,p+1)) / 10**11

havier_garcia = np.zeros((p+1,p+1))


for i, part in enumerate(parts):
    data[part[0], part[1]] = np.abs(a[i])
    havier_garcia[part[0], part[1]] = np.sqrt(cov[i,i]) / np.abs(a[i])
data = data[::-1,:]
havier_garcia = havier_garcia[::-1,:]

np.set_printoptions(linewidth=500)

for i in range(len(data[:,0])):
    print(havier_garcia[i,:])


# create discrete colormap
cmap = cm.get_cmap("gist_gray")
norm = colors.LogNorm(10**-11, max(a))

plt.imshow(data, cmap=cmap, norm=norm, extent=[0,p+1,0,p+1])

# draw gridlines
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

plt.xticks([i for i in range(0,p+1,1)], ["$x_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])
plt.yticks([i for i in range(0,p+1,1)], ["$\\theta_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])

plt.colorbar(label = "$\\chi^2$")

plt.show()


# create discrete colormap
cmap = cm.get_cmap("gist_gray")
norm = colors.Normalize(-18, 11)

plt.imshow(np.log10(havier_garcia), cmap=cmap, norm=norm, extent=[0,p+1,0,p+1])

# draw gridlines
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

plt.xticks([i for i in range(0,p+1,1)], ["$x_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])
plt.yticks([i for i in range(0,p+1,1)], ["$\\theta_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])

plt.colorbar(label = "log$_{10}$(Relativna napaka)")

plt.show()






