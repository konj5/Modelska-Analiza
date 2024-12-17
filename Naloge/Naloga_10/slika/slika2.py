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

image = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\lena_k1_n0.pgm')

kernel = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\kernel1.pgm')

from skimage import color, data, restoration


#deconvolved, _ = restoration.unsupervised_wiener(image, kernel)
deconvolved= restoration.wiener(image, kernel, balance=5000000)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)

plt.gray()

ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Dekonvolucija')

fig.tight_layout()

plt.show()