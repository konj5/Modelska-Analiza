import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)


from deconvolve_lib import load_image, deconvolve_pure, deconvolve_wiener, deconvolve_wiener_window, extend, retract, readpgm


base_image = image = readpgm(f'Naloge\\Naloga_10\\slika\\lena_slike\\lena_k3_n0.pgm')
image = readpgm(f'Naloge\\Naloga_10\\slika\\lena_slike\\lena_k3_nx.pgm')
kernel = readpgm(f'Naloge\\Naloga_10\\slika\\lena_slike\\kernel3.pgm')


"""fig, axs = plt.subplots(1,2)
ax1, ax2 = axs
fig.set_figheight(3)
fig.set_figwidth(6)

FFT_base = np.abs(np.fft.fft2(base_image))
FFT = np.abs(np.fft.fft2(image))



ax1.imshow(FFT_base, cmap = "binary_r", norm = colors.LogNorm())
ax1.set_title("brez motnje")
ax1.axis("off")
ax2.imshow(FFT, cmap = "binary_r", norm = colors.LogNorm())
ax2.set_title("z motnjo")
ax2.axis("off")
plt.show()"""

def remove_peaks_bad(image):
    fspec = np.fft.fft2(image)

    for i in range(fspec.shape[0]):
        for j in range(fspec.shape[1]):
            fspec[i,j] -= np.sum(fspec[max(0,i-1):min(i+1, fspec.shape[0]-1), j]) + np.sum(fspec[i, max(0,j-1):min(j+1, fspec.shape[0]-1)])
            fspec[i,j] *= -1/5

    return np.fft.ifft2(fspec)

def remove_peaks(image):
    fspec = np.fft.fft2(image)
    n = 0
    N = 0
    for i in range(fspec.shape[0]):
        for j in range(fspec.shape[1]):
            a = 1
            N += 1
            if np.abs(fspec[i,j]) > 10**6:
                n += 2
                fspec[max(0,i-a):min(i+a, fspec.shape[0]-1),max(0,j-a):min(j+a, fspec.shape[0]-1)] = np.average(fspec[max(0,i-2*a):min(i+2*a, fspec.shape[0]-1),max(0,j-2*a):min(j+2*a, fspec.shape[0]-1)])

    print(N)
    print(n)


    return np.fft.ifft2(fspec)

def remove_peaks_cringe(image):
    points = [[51,2], [65,7], [98,9],[103,4], [130,14], [153,6], [164,17], [195,20], [206,8], [299,23]]
    temp = [[4,32], [7,64], [11,96], [14,128], [18,160], [22,190], [26,223]]; points.extend(temp)
    temp = [[32,286], [35,318], [19,360], [15,390], [11,421], [8,450], [4,482]]; points.extend(temp)

    fspec = np.fft.fft2(image)

    for point in points:
        #print(point)
        j,i = point
        a = 5

        fspec[max(0,i-a):min(i+a, fspec.shape[0]-1),max(0,j-a):min(j+a, fspec.shape[0]-1)] = np.average(fspec[max(0,i-6*a):min(i+6*a, fspec.shape[0]-1),max(0,j-6*a):min(j+6*a, fspec.shape[0]-1)])

        j,i = fspec.shape[0]-j, fspec.shape[0]-i

        fspec[max(0,i-a):min(i+a, fspec.shape[0]-1),max(0,j-a):min(j+a, fspec.shape[0]-1)] = np.average(fspec[max(0,i-6*a):min(i+6*a, fspec.shape[0]-1),max(0,j-6*a):min(j+6*a, fspec.shape[0]-1)])


    return fspec



fig, axs = plt.subplots(1,2)
ax1, ax2 = axs
fig.set_figheight(3)
fig.set_figwidth(6)
ax1.imshow(np.abs(np.fft.fft2(image)), cmap = "binary_r", norm = colors.LogNorm(np.min(np.abs(np.fft.fft2(image))), np.max(np.abs(np.fft.fft2(image)))))
ax1.set_title("z motnjo")
ax1.axis("off")
ax2.imshow(np.abs(remove_peaks_cringe(image)), cmap = "binary_r", norm = colors.LogNorm(np.min(np.abs(np.fft.fft2(image))), np.max(np.abs(np.fft.fft2(image)))))
ax2.set_title("odstranjena motnja")
ax2.axis("off")
plt.show()

for k in [2]:
    for n in [2]:
        fig, axs = plt.subplots(1,4)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        ax1, ax2, ax3, ax4 = axs

        nimage = image.copy()
        ax1.imshow(nimage, cmap="binary_r")
        ax1.set_title("Začetna slika")
        ax1.axis('off')

        nimage = deconvolve_wiener_window(extend(image), kernel, 1000000)
        ax2.imshow(retract(nimage), cmap="binary_r")
        ax2.set_title("samo filtrirana")    
        ax2.axis('off')

        nimage = remove_peaks_cringe(image)
        ax3.imshow(np.abs(np.fft.ifft2(nimage)), cmap="binary_r")
        ax3.set_title("brez motnje")
        ax3.axis('off')


        nimage = deconvolve_wiener_window(extend(np.abs(np.fft.ifft2(remove_peaks_cringe(image)))), kernel, 1000000)
        ax4.imshow(retract(nimage), cmap="binary_r")
        ax4.set_title("filtrirana brez motnje")
        ax4.axis('off')

plt.show()
