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

def kalman_iter_no_measure(x0, P0, z):
    xbar = F.dot(x0) + c
    Pbar = F.dot(P0.dot(F.T)) + Q

    xnew = xbar
    Pnew = Pbar

    return xnew, Pnew

def verr(vx,vy):
    return max(0.278, 0.01 * np.sqrt(vx**2 + vy**2))


fig, axs = plt.subplots(1,2)
ax3,ax4 = axs

import seaborn as sns
palette = sns.color_palette(n_colors=10)

for j, skipn in enumerate([100,10,1]):
    t,x,y,vx,vy,ax,ay = data[0,:]
    xs, Ps = np.array([x,y,vx,vy]), np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

    xss = np.zeros((data.shape[0], 4))
    Pss = np.zeros((data.shape[0], 4, 4))

    xss[0,:] = xs
    #print(xss[0,:])
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

        if i % skipn == 0:
            xs, Ps = kalman_iter(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
            print(1)
        else:
            xs, Ps = kalman_iter_no_measure(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
            print(0)

        xss[i,:] = xs
        Pss[i,:,:] = Ps


    #print(xss[0,:])
    #print(kontrola[0,:])


    

    ax3.scatter(kontrola[:,1]-xss[:,0], kontrola[:,2]-xss[:,1], c = palette[j], s = 2)
    ax4.scatter(kontrola[:,3]-xss[:,2], kontrola[:,4]-xss[:,3], c = palette[j], s = 2, label = skipn)
    



ax3.set_xlabel("$x_{\\text{kontrola}} - x$")
ax3.set_ylabel("$y_{\\text{kontrola}} - y$")

ax4.set_xlabel("$v_{x,\\text{kontrola}} - v_x$")
ax4.set_ylabel("$v_{y,\\text{kontrola}} - v_y$")


    ######

    #plt.plot(kontrola[::skipn,0], np.sqrt((kontrola[::skipn,1]-xss[:len(xss)//skipn,0])**2 + (kontrola[::skipn,2]-xss[:len(xss)//skipn,1])**2), label = f"{skipn}")

ax4.legend()
plt.show()


