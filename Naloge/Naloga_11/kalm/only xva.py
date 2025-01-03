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

t,x,y,vx,vy,ax,ay = data[0,:]
xs, Ps = np.array([x,y,vx,vy]), np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

xss = np.zeros((data.shape[0], 4))
Pss = np.zeros((data.shape[0], 4, 4))

xss[0,:] = xs
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

    xs, Ps = kalman_iter_xa(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
    xss[i,:] = xs
    Pss[i,:,:] = Ps

xss_xa = np.copy(xss)
Pss_xa = np.copy(Pss)

#######################

t,x,y,vx,vy,ax,ay = data[0,:]
xs, Ps = np.array([x,y,vx,vy]), np.diag([25**2, 25**2, verr(vx,vy)**2, verr(vx,vy)**2])

xss = np.zeros((data.shape[0], 4))
Pss = np.zeros((data.shape[0], 4, 4))

xss[0,:] = xs
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

    xs, Ps = kalman_iter_v(xss[i-1,:], Pss[i-1,:,:], np.array([x,y,vx,vy]))
    xss[i,:] = xs
    Pss[i,:,:] = Ps

xss_v = np.copy(xss)
Pss_v = np.copy(Pss)

########

plt.plot(xss_xa[:,0], xss_xa[:,1], label = "Samo pozicija in pospešek")
plt.plot(xss_v[:,0], xss_v[:,1], label = "Samo hitrost")
plt.plot(kontrola[:,1], kontrola[:,2], color = "black", label = "Kontrola")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()


########

plt.plot(xss_xa[:,0], xss_xa[:,1], label = "Samo pozicija in pospešek")
plt.plot(xss_v[:,0], xss_v[:,1], label = "Samo hitrost")
plt.plot(kontrola[:,1], kontrola[:,2], color = "black", label = "Kontrola")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()