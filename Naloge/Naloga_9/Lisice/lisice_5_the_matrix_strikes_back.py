import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)




Nmat = int(6)
M = np.zeros((Nmat, Nmat, Nmat, Nmat))


alpha = 1
beta = 1
dt = 0.01
tmax = 10

Z0 = 200
L0 = 50

for i in range(Nmat):
    for j in range(Nmat):
        for k in range(Nmat):
            for l in range(Nmat):
                A = 5*alpha*k*dt
                B = 4*alpha*k*dt
                C = alpha/L0*k*l*dt

                D = 4*beta*l*dt
                E = 5*beta*l*dt
                F = beta/Z0*k*l*dt

                if i == k and j == l:
                    M[i,j,k,l] += 1-A-B-C-D-E-F

                if i == k+1 and j == l:
                    M[i,j,k,l] += A

                if i == k-1 and j == l:
                    M[i,j,k,l] += B + C

                if i == k and j == l+1:
                    M[i,j,k,l] += D

                if i == k and j == l-1:
                    M[i,j,k,l] += E + F


x = np.zeros((Nmat,Nmat)); x[3,3] = 1



import matplotlib.animation as animation


fps = 30


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,10) )

cmap = cm.get_cmap("hot")
norm = colors.LogNorm(10**-5, 1)

im = plt.imshow(x,interpolation='none', cmap=cmap, norm=norm, extent=[0,Nmat,0,Nmat], origin="lower", aspect='auto')
plt.colorbar()

def animate_func(i):
    global x
    if i % fps == 0:
        print( '              ', end ='\r' )
        print( f'{i * dt:0.1f} / {tmax}', end ='\r' )

    x = np.einsum("ijkl, kl -> ij", M, x)
    x = x / np.sum(x)

    im.set_array(x)
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = int(tmax/dt),
                               interval = 1000 / fps, # in ms
                               )

anim.save('test_anim.mp4', fps=fps)

print('Done!')

                

