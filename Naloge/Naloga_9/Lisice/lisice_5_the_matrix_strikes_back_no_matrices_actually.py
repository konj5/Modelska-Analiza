import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

from scipy import sparse 


Nmat = int(800)



alpha = 1
beta = 1
dt = 0.001
tmax = 10

Z0 = 200
L0 = 50


def Matmultiply(x):
    x_new = np.zeros_like(x)
    for i in range(Nmat):
        for j in range(Nmat):
            x_new[i,j] = x[i,j] * (1-5*alpha*i*dt-4*alpha*i*dt-alpha/L0*i*j*dt-4*beta*j*dt-5*beta*j*dt-beta/Z0*i*j*dt)
            
            if i != Nmat-1:
                x_new[i,j] += x[i+1,j] * (4*alpha*(i+1)*dt+alpha/L0*(i+1)*j*dt)
            
            if j != Nmat-1:
                 x_new[i,j] += x[i,j+1] * (5*beta*(j+1)*dt+beta/Z0*i*(j+1)*dt)

            if i != 0:
                x_new[i,j] += x[i-1,j] * (5*alpha*(i-1)*dt)
            
            if j != 0:
                x_new[i,j] += x[i,j-1] * (4*beta*(j-1)*dt)
    return x_new

    
def Matmultiply_N(x,N):
    for i in range(N):
        x = Matmultiply(x)
        x = x / np.sum(x)

    return x



x = np.zeros((Nmat,Nmat)); 
x[200,50] = 1
#x[3,3]=1


import matplotlib.animation as animation


fps = 30


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,10) )

cmap = cm.get_cmap("hot")
norm = colors.LogNorm(10**-5, 1)

im = plt.imshow(x,interpolation='none', cmap=cmap, norm=norm, extent=[0,Nmat,0,Nmat], origin="lower", aspect='auto')
plt.colorbar()

Nts = int(tmax/dt)
time = 30
Nframes = time * fps
Nskip = max(1, int(Nts/Nframes))
print(Nskip)

import time
realstime = time.time()
def animate_func(i):
    global x

    print( '                                  ', end ='\r' )
    print( f'{i}, {i * dt:0.1f} / {tmax}, elapsed: {(time.time()-realstime)/60:0.1f} min  remaining:{(tmax-i*dt)/dt * (time.time()-realstime)/60 /(i+1):0.1f} min', end ='\r' )
    
    x = Matmultiply_N(x,Nskip)

    im.set_array(x)
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = int(tmax/dt),
                               )

anim.save('test_anim.mp4', fps=fps)

print('Done!')

                

