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



def f(x,t,c="Ligma balčs"):
    def C(i):
        if c is None: return 1
        else:
            return c[i]

    return C(0) + C(1) * t + C(2) * x + C(3) * t**2 + C(4) * x*t + C(5) * x**2 + C(6) * x**2 * t  + C(7) * x * t**2 + C(8) * t**3

parts = [[0,0], [0,1], [1,0], [2,0], [1,1], [0,2], [2,1], [1,2], [0,3]]

#Define matrix
N = len(ttg); M = 9; A = np.zeros((N,M))
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
print(hi2)

plt.show()

## plot chi^2 by q

##plot coefitients for q = 5 in grid with color gradient. Remove the ones that dont contribute!

### Reši z kalmanom (problem je linearen!)


data = np.ones((p+1,p+1)) / 10**11

print(data.shape)

for i, part in enumerate(parts):
    data[part[0], part[1]] = np.abs(a[i])
data = data[::-1,:]


for i in range(len(data[:,0])):
    print(data[i,:])


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

