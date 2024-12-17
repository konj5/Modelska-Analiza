import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)

"""def readpgm(name):
    with open(name) as f:
         lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
            if l[0] == '#':
                lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    return (np.array(data[3:]),(data[1],data[0]),data[2])"""

def readpgm(name):
    with open(name) as f:
         lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
            if l[0] == '#':
                lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2'

    # Converts data to a list of integers
    data = np.zeros((512,512))
    lindata = []
    for line in lines[3:]:
        lindata.extend([int(c) for c in line.split()])

    for i in range(len(lindata)):
        data[i//512,i%512] = lindata[i]

    return data

data = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\lena_k1_n0.pgm')

kernel = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\kernel1.pgm')

"""print(data.shape)
print(kernel.shape)

plt.imshow(data, cmap="binary_r", norm=colors.Normalize(0,255))
plt.show()

plt.imshow(kernel, cmap="binary_r", norm=colors.Normalize(0,255))
plt.show()"""

plt.imshow(kernel[200:-200,200:-200], cmap="binary_r", norm=colors.Normalize(0,255))
plt.axis("off")
plt.show()

def deconcolution_noiseless(data):
    dfr = np.fft.fft2(data)
    return np.real(np.fft.ifftshift(np.fft.ifft2(dfr/np.fft.fft2(kernel))))

fig, axs = plt.subplots(1,2); ax1, ax2 = axs

decon = deconcolution_noiseless(data)

ax1.imshow(data, cmap="binary_r")
ax2.imshow(decon, cmap="binary_r")

plt.show()