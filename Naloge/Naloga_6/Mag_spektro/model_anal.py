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


model1 = [[0,0], [0,1], [1,0]]

model2 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0]]
model3 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [1,2], [2,1], [3,0]]

model4 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [1,2], [2,1], [3,0], [0,3], [1,3], [2,2], [4,0]]

model2_1 = [[0,0], [0,1], [1,0], [0,2], [2,0]]
#model2_1 = [[0,0], [0,1], [1,0], [0,2], [1,1]]

model3_1 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [1,2], [2,1]]
model3_2 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [2,1], [3,0]]
model3_3 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [2,1]]

model3_4 = [[0,0], [0,1], [1,0], [0,2], [1,1], [2,0], [2,1]]

parts = model1

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
U, s, Vh = svd(H)

a = np.zeros(M)
for i in range(M):
    a += U[:,i].dot(b)/s[i]  * Vh[i,:]

cov = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        for k in range(M):
            cov[i,j] += Vh[k,i] * Vh[k,j] / s[k]**2

print(cov)


#Plot Stuff

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
hi2 = (H.dot(a)-b).dot(H.dot(a)-b)
print(f"$\\chi^2 =$ {hi2}")

plt.show()


#######

p = np.max(np.array(parts))

data = np.zeros((p+1,p+1)) / 10**11

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
cmap = cm.get_cmap("cool")
norm = colors.LogNorm(min(a[a>0]), max(a))


plt.imshow(data, cmap=cmap, norm=norm, extent=[0,p+1,0,p+1])

# draw gridlines
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(i+0.3, j+0.5, f"{data[i,j]:0.3e}", color='black')

plt.xticks([i for i in range(0,p+1,1)], ["$x_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])
plt.yticks([i for i in range(0,p+1,1)], ["$\\theta_{fp}$"+f"$^{i}$" for i in range(0,p+1,1)])

plt.colorbar(label = "$c_i$")

plt.show()

####

p = len(parts)

temp = np.zeros_like(cov)
for i in range(cov.shape[0]):
    for j in range(cov.shape[0]):
        temp[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
cov = temp

cmap = cm.get_cmap("plasma")
norm = colors.Normalize(0, np.max(np.abs(cov)))
plt.imshow(np.abs(cov), cmap=cmap, norm=norm, extent=[0,p,0,p])
cov = cov[:,::-1]



for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        plt.text(i+0.1, j+0.5, f"{cov[i,j]:0.3f}", color='black')

# draw gridlines
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

labels = ["$x_{}^{} \\cdot \\theta_{}^{}$".format("{fp}", part[0],"{fp}", part[1]) for part in parts]

plt.xticks([i for i in range(0,p,1)], labels)
plt.yticks([i for i in range(0,p,1)], labels[::-1])

plt.colorbar(label = "$|\\rho|$")

plt.show()
