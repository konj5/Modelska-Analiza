import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

with open("Naloge\\Naloga_11\\kalm\\kalman_relative_data.dat", mode = "r") as f:
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

def kalman_iter_nox(x0, P0, z):
    xbar = F.dot(x0) + c
    Pbar = F.dot(P0.dot(F.T)) + Q

    H[0,0] = 0; H[1,1] = 0

    K = Pbar.dot(H.T).dot(np.linalg.inv(H.dot(Pbar).dot(H.T) + R))
    xnew = xbar + K.dot(z-H.dot(xbar))
    Pnew = Pbar - K.dot(H).dot(Pbar)

    return xnew, Pnew


def kalman_iter_xa(x0, P0, z):
    xbar = F.dot(x0) + c
    Pbar = F.dot(P0.dot(F.T)) + Q

    H[2,2] = 0; H[3,3] = 0; 

    K = Pbar.dot(H.T).dot(np.linalg.inv(H.dot(Pbar).dot(H.T) + R))
    xnew = xbar + K.dot(z-H.dot(xbar))
    Pnew = Pbar - K.dot(H).dot(Pbar)

    return xnew, Pnew

def kalman_iter_v(x0, P0, z):
    xbar = F.dot(x0) + c
    Pbar = F.dot(P0.dot(F.T)) + Q

    H[0,0] = 0; H[1,1] = 0 

    K = Pbar.dot(H.T).dot(np.linalg.inv(H.dot(Pbar).dot(H.T) + R))
    xnew = xbar + K.dot(z-H.dot(xbar))
    Pnew = Pbar - K.dot(H).dot(Pbar)

    return xnew, Pnew

def verr(vx,vy):
    return max(0.278, 0.01 * np.sqrt(vx**2 + vy**2))

def Bmat(vx,vy):
    if vx == 0 and vy == 0: return np.array([[1,-1], [1,1]])
    else:
        return 1/np.linalg.norm([vx,vy]) * np.array([[vx,-vy],[vy,vx]])

t,x,y,at,ar = data[0,:]

vx, vy = 0, 0#########################

xs, Ps = np.array([x,y,vx,vy]), np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

xss = np.zeros((data.shape[0], 4))
Pss = np.zeros((data.shape[0], 4, 4))

xss[0,:] = xs
Pss[0,:,:] = Ps

xsss = []
Psss = []

skipns = [1,5,10]

for skipn in skipns:

    for i in tqdm(range(1, data.shape[0])):
        tprev = data[i-1,0]
        t,x,y,at,ar = data[i,:]

        _,_,vx,vy = xss[i-1,:]

        dt = t-tprev

        Bm = Bmat(vx,vy)
        B = np.eye(4)
        B[2:,2:] = Bm

        F = getF(dt)
        c = B.dot(np.array([0,0,at,ar]) * dt)


        Q = np.diag([0,0,0.05**2 * dt**2, 0.05**2 * dt**2])

        Pvv = Pss[i-1, 2:,2:]
        v = np.array([vx,vy])
        a = np.array([at,ar])
        if np.linalg.norm(v) == 0:
            qa = 0
        else:
            qa =v.dot(Pvv.dot(v))/np.linalg.norm(v)**4 * (np.outer(Bm.dot(a),Bm.dot(a)))

        Q[2:,2:] += qa


        R = np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

        H = np.eye(4)

        if i % skipn == 0:
            xs, Ps = kalman_iter(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
        else:
            xs, Ps = kalman_iter_nox(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))

        xss[i,:] = xs
        Pss[i,:,:] = Ps

    xss_xa = np.copy(xss)
    Pss_xa = np.copy(Pss)

    xsss.append(xss_xa)
    Psss.append(Pss_xa)




########

for i,xss_xa in enumerate(xsss):
    plt.plot(xss_xa[:,0], xss_xa[:,1], label = f"k = {skipns[i]}")
plt.plot(kontrola[:,1], kontrola[:,2], color = "black", label = "Kontrola")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()


########

for i,xss_xa in enumerate(xsss):
    plt.plot(xss_xa[:,2], xss_xa[:,3], label = f"k = {skipns[i]}")
plt.plot(kontrola[:,3], kontrola[:,4], color = "black", label = "Kontrola")
plt.xlabel("$v_x$")
plt.ylabel("$v_y$")
plt.legend()

plt.show()


#######

fig, axs = plt.subplots(1,2); ax1,ax2=axs

for i,xss_xa in enumerate(xsss):

    ax1.scatter(kontrola[:,1]-xss_xa[:,0], kontrola[:,2]-xss_xa[:,1], s = 2, label = f"k = {skipns[i]}")

    ax2.scatter(kontrola[:,3]-xss_xa[:,2], kontrola[:,4]-xss_xa[:,3], s = 2, label = f"k = {skipns[i]}")

ax1.set_xlabel("$x_{\\text{kontrola}} - x$")
ax1.set_ylabel("$y_{\\text{kontrola}} - y$")

ax2.set_xlabel("$v_{x,\\text{kontrola}} - v_x$")
ax2.set_ylabel("$v_{y,\\text{kontrola}} - v_y$")

ax2.legend()

plt.show()

#######


fig, axs = plt.subplots(1,1); ax1=axs

for i,Pss_xa in enumerate(Psss):

    ax1.plot(kontrola[:,0], np.sqrt(Pss_xa[:,0,0]**2 + (Pss_xa[:,1,1]**2)), label = f"k = {skipns[i]}")

    ax2.plot(kontrola[:,0], np.sqrt(Pss_xa[:,2,2]**2 + (Pss_xa[:,3,3]**2)), label = f"k = {skipns[i]}")

ax1.set_xlabel("čas")
ax1.set_ylabel("$\\sqrt{\\sigma_x^2+\\sigma_y^2}$")
ax1.set_yscale("log")

ax1.legend()

plt.show()