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

#image = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\lena_k1_n0.pgm')

#kernel = readpgm('Naloge\\Naloga_10\\slika\\lena_slike\\kernel1.pgm')

from skimage import color, data, restoration


#deconvolved, _ = restoration.unsupervised_wiener(image, kernel)
#deconvolved= restoration.wiener(image, kernel, balance=5000000)

def load_image(k,n):
    image = readpgm(f'Naloge\\Naloga_10\\slika\\lena_slike\\lena_k{k}_n{n}.pgm')
    kernel = readpgm(f'Naloge\\Naloga_10\\slika\\lena_slike\\kernel{k}.pgm')
    return image, kernel

def deconvolve_pure(image, kernel):
    dfr = np.fft.fft2(image)
    return np.real(np.fft.ifftshift(np.fft.ifft2(dfr/np.fft.fft2(kernel))))

def deconvolve_wiener(image, kernel, balance):
    return restoration.wiener(image, kernel, balance=balance)
    return restoration.unsupervised_wiener(image, kernel)[0]

def based_window_base(n):
    window_base = np.ones(n)

    for i in range(n//10):
        window_base[i] = i/(n//10)
        window_base[-i-1] = window_base[i]
    
    return window_base


def deconvolve_wiener_window(image, kernel, balance):
    #window_base = np.hanning(image.shape[0])
    window_base = based_window_base(image.shape[0])
    window = np.zeros(image.shape)

    for i in range(image.shape[0]):
        window[:,i] = window_base

    for i in range(image.shape[0]):
        window[i,:] = window[i,:] * window_base

    return restoration.wiener(image * window, kernel, balance=balance)
    return restoration.unsupervised_wiener(image * window, kernel)[0]

def extend(image):
    N = image.shape[0]
    nimage = np.zeros((3*N,3*N))

    nimage[N:2*N, N:2*N] = image

    for i in range(image.shape[0]):
        nimage[N+i,:N] = nimage[N+i,N]
        nimage[N+i,2*N:] = nimage[N+i,2*N-1]

        nimage[:N, N+i,] = nimage[N,N+i]
        nimage[2*N:,N+i] = nimage[2*N-1, N+i]

    nimage[0:N,0:N] = nimage[N,N]
    nimage[0:N,2*N:3*N] = nimage[N,2*N-1]
    nimage[2*N:3*N,0:N] = nimage[2*N-1,N]
    nimage[2*N:3*N,2*N:3*N] = nimage[2*N-1,2*N-1]
    
    return nimage

def retract(image):
    N = image.shape[0]//3
    return image[N:2*N, N:2*N]


"""image, kernel = load_image(2,0)
#nimage = deconvolve_wiener_window(image, kernel, 5*10**7)

#nimage = extend(image)
#nimage = deconvolve_wiener_window(nimage, kernel, 5*10**7)

plt.imshow(nimage, cmap="binary_r")
plt.axis("off")
plt.show()
"""

