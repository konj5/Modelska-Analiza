import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

with open("Naloge\\Naloga_11\\kalm\\kalman_cartesian_data.dat", mode = "r") as f:
    lines = f.readlines()
    data = np.zeros((len(lines), len(lines[0].split(" "))))
    for i, line in enumerate(lines):
        for j, x in enumerate(line.split(" ")):
            data[i,j] = np.float32(x)

with open("Naloge\\Naloga_11\\kalm\\kalman_cartesian_kontrola.dat", mode = "r") as f:
    lines = f.readlines()
    kontrola = np.zeros((len(lines), len(lines[0].split(" "))))
    for i, line in enumerate(lines):
        for j, x in enumerate(line.split(" ")):
            kontrola[i,j] = np.float32(x)

def getF(dt):
    F = np.eye(4)
    F[:2,2:] = np.eye(2) * dt

    return F


def kalman_iter(x0, P0, z):
    xbar = F.dot(x0) + c
    Pbar = F.dot(P0.dot(F.T)) + Q

    K = Pbar.dot(H.T).dot(np.linalg.inv(H.dot(Pbar).dot(H.T) + R))
    xnew = xbar + K.dot(z-H.dot(xbar))
    Pnew = Pbar - K.dot(H).dot(Pbar)

    return xnew, Pnew

def verr(vx,vy):
    return max(0.278, 0.01 * np.sqrt(vx**2 + vy**2))

t,x,y,vx,vy,ax,ay = data[0,:]
xs, Ps = np.array([x,y,vx,vy]), np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

xss = np.zeros((data.shape[0], 4))
Pss = np.zeros((data.shape[0], 4, 4))

xss[0,:] = xs
print(xss[0,:])
Pss[0,:,:] = Ps

for i in tqdm(range(1, data.shape[0])):
    tprev = data[i-1,0]
    t,x,y,vx,vy,ax,ay = data[i,:]
    dt = t-tprev

    F = getF(dt)
    c = np.array([0,0,ax*dt, ay*dt])
    Q = np.diag([0,0,0.05**2 * dt**2, 0.05**2 * dt**2])
    R = np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

    H = np.eye(4)

    xs, Ps = kalman_iter(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
    xss[i,:] = xs
    Pss[i,:,:] = Ps


print(xss[0,:])
print(kontrola[0,:])


plt.scatter(xss[:,0], xss[:,1], s = 4, color = "black", label = "Filtrirani rezultati")
plt.plot(kontrola[:,1], kontrola[:,2], label = "Točni rezultat")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()


######

plt.plot(kontrola[:,0], np.sqrt((kontrola[:,1]-xss[:,0])**2 + (kontrola[:,2]-xss[:,1])**2))

plt.show()

####

plt.plot(kontrola[:,0], np.sqrt(Pss[:,1,1] + Pss[:,2,2]))
plt.xlabel("čas")
plt.ylabel("$\\sqrt{\\sigma_x^2 + \\sigma_y^2}$")
plt.yscale("log")

plt.show()


######
######

plt.plot(kontrola[:,0], xss[:,2], label = "filtrirana")
plt.plot(kontrola[:,0], kontrola[:,3], label = "kontrola")

plt.xlabel("čas")
plt.ylabel("$v_x$")
plt.legend()

plt.show()

######

plt.plot(kontrola[:,0], xss[:,3], label = "filtrirana")
plt.plot(kontrola[:,0], kontrola[:,4], label = "kontrola")

plt.xlabel("čas")
plt.ylabel("$v_y$")
plt.legend()

plt.show()

######

plt.plot(kontrola[:,0], xss[:,2], label = "filtrirana $v_x$")
plt.plot(kontrola[:,0], kontrola[:,3], label = "točna $v_x$")
plt.plot(kontrola[:,0], xss[:,3], label = "filtrirana $v_y$")
plt.plot(kontrola[:,0], kontrola[:,4], label = "točna $v_y$")
plt.xlabel("čas")
plt.ylabel("hitrost")
plt.yscale("log")

plt.show()
