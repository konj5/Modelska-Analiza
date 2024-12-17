import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re, sys
from scipy.integrate import solve_ivp
np.set_printoptions(linewidth=10000, threshold=sys.maxsize)


from deconvolve_lib import load_image, deconvolve_pure, deconvolve_wiener, deconvolve_wiener_window, extend, retract

kstit = ["tresoč objektiv", "slab fokus", "uklonska mrežica"]

for sigma in [0.5,1,5,10,40]:
    for n in [0,4,8,16]:
        fig, axs = plt.subplots(1,4)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        ax1, ax2, ax3, ax4 = axs
        image, kernel = load_image(2,n)

        kernel = np.zeros_like(kernel)
        N = kernel.shape[0]
        for i in range(N):
            for j in range(N):
                kernel[i,j] = np.round(255 * np.exp(-((i-N//2)**2 + (j-N//2)**2)/(2*sigma**2)))


        image = extend(image); kernel = extend(kernel)
        fig.suptitle(f"$\\sigma = {sigma}$, RMS Gaussovega šuma {n}")

        nimage = image.copy()
        ax1.imshow(retract(nimage), cmap="binary_r", norm = colors.Normalize(0, 255))
        ax1.set_title("Začetna slika")
        ax1.axis('off')


        nimage = deconvolve_pure(image, kernel)
        ax2.imshow(retract(nimage), cmap="binary_r")
        ax2.set_title("Dekonvolucija")
        ax2.axis('off')

        nimage = deconvolve_wiener(image, kernel, 1000000)
        ax3.imshow(retract(nimage), cmap="binary_r")
        ax3.set_title("še z Wienerjevim filtrom")
        ax3.axis('off')

        nimage = deconvolve_wiener_window(image, kernel, 1000000)
        ax4.imshow(retract(nimage), cmap="binary_r")
        ax4.set_title("še z oknom")
        ax4.axis('off')

        plt.savefig(f"sigma{sigma}, n{n} extended.png")
        plt.show()