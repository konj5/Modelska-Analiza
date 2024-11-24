import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

# ANALITIČEN FIT SIMPEL MODEL 

fig, axs = plt.subplots(1,2)

ax1, ax2 = axs

with open("Naloge\\Naloga_6\\farmakoloski.dat", mode = "r") as f:
    f.readline()
    lines = f.readlines()

#Extract data
xs, ys, dys = [], [], []
for line in lines:
    line = line.strip()
    split = re.split("\t+", line)
    xs.append(float(split[0])); ys.append(float(split[1])); dys.append(float(split[2]))
xs = np.array(xs); ys = np.array(ys); dys = np.array(dys)

#Reformat data to analytic fit
xs = xs
fs = 1/ys
dfs = dys/ys**2


#solve
from scipy.optimize import least_squares

def F(c):
    c0, c1 = c

    return (c0 + c1/xs - fs) / dfs

sol = least_squares(F, x0=[1,1], method='lm')


c = sol.x

J = sol.jac
M = np.linalg.inv(J.T.dot(J))
print(c)

ax1.errorbar(xs, ys, dys, color='red', ls='', marker='o', capsize=5, capthick=1, ecolor='black')

def f(x):
    return c[0] * 1 + c[1] * 1/x

new_xs = np.linspace(min(xs), max(xs), 1000)
new_ys = 1/f(new_xs)
ax1.plot(new_xs, new_ys)

#hi2 = np.sum((ys-1/f(xs))**2 / dys**2)
y0 = 1/c[0]
a = c[1] * y0
J = np.zeros([2,2])
J[0,0] = -1/c[0]**2; J[1,0] = -c[1]/c[0]**2; J[1,1] = 1/c[0]

M = J.dot(M.dot(J.T))

def f(x):
    return y0*x/(x + a) 

hi2 = np.sum((ys-f(xs))**2 / dys**2)

ax1.set_title(f"Numerično prilagajanje: $\\chi^2 = {hi2:0.2f}$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")



#Confidence interval

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip




plot_cov_ellipse(M,[y0, a],ax=ax2, color = "blue", alpha = 0.3, nstd=2, label = "95%")
plot_cov_ellipse(M,[y0, a],ax=ax2, color = "green", alpha = 0.3, nstd=1, label = "68%")

ax2.scatter(y0, a, c = "black")

ax2.set_title(f"Interval zaupanja:  $y_0 = {y0:0.0f}\\pm{np.sqrt(M[0,0]):0.2f}$, $a={a:0.1f}\\pm{np.sqrt(M[1,1]):0.3f}$ ")

ax2.set_ylim(a-2*np.sqrt(M[1,1]), a+2*np.sqrt(M[1,1]))
ax2.set_xlim(y0-2*np.sqrt(M[0,0]), y0+2*np.sqrt(M[0,0]))
ax2.legend()

ax2.set_xlabel("$y_0$")
ax2.set_ylabel("$a$")

#plt.tight_layout()

plt.show()











