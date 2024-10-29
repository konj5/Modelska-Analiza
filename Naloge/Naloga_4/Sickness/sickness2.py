import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta = args

    vdot[0] = -alpha * v[0] * v[1]
    vdot[1] = alpha * v[0] * v[1] - beta * v[1]
    vdot[2] = beta * v[1]
    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/1000)
    return sol.t, sol.y

startstate = np.array([0.99,0.01,0])

alphas = np.linspace(0,2,100)
betas = np.linspace(0,2,100)

xdata, ydata, bmax, tmax, total = [], [], [], [], []
for i in tqdm(range(len(alphas)), desc = "x"):
    for j in tqdm(range(len(betas)), desc = "y", leave= False):
        alpha, beta = alphas[i], betas[j]

        xdata.append(alpha)
        ydata.append(beta)

        ts, vs = solve(startstate, f, [alpha, beta], 0, 100)


        bmax.append(np.max(vs[1,:]))
        tmax.append(ts[np.argmax(vs[1,:])])
        total.append(vs[1,-1] + vs[2,-1])


        
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.

# Plot the surface.

surf = ax.plot_trisurf(xdata, ydata, bmax, cmap=cm.coolwarm)

ax.set_xlabel("$\\alpha$")
ax.set_ylabel("$\\beta$")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.

# Plot the surface.

surf = ax.plot_trisurf(xdata, ydata, tmax, cmap=cm.coolwarm)

ax.set_xlabel("$\\alpha$")
ax.set_ylabel("$\\beta$")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.

# Plot the surface.

surf = ax.plot_trisurf(xdata, ydata, total, cmap=cm.coolwarm)

ax.set_xlabel("$\\alpha$")
ax.set_ylabel("$\\beta$")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()