import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re
from monte_carlo import monte_carlo

"""

V duhu naloge nisem hotel uporabljani preveč analitičnih trikov, saj teh v veliko primerih nebi mogli uporabiti in tak primer zato
nebi bil reprezentativen za uporabo metode monte carlo na splošno.


"""





def rho_r(x):
    r, theta, phi = x
    return r**p

def rho_x(x):
    x, y, z = x
    return (x**2 + y**2 + z**2)**(p/2)

def constraint_r(x):
    r, theta, phi = x
    x, y, z = r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)

    return 1 if np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1 else 0

def constraint_x(x):
    x, y, z = x

    return 1 if np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1 else 0

limits_x = [[-1,1], [-1,1], [-1,1]]

limits_r = [[0,1], [0,np.pi], [0,2*np.pi]]


from scipy.integrate import tplquad
def comparison_integrator(f:callable):
    a,b = [0,1]
    
    q = 0
    r = lambda x,y: (1-np.sqrt(x)-np.sqrt(y))**2

    g = 0
    h = lambda x: (1-np.sqrt(x))**2

    return tplquad(lambda x1,x2,x3: 8 * f([x1,x2,x3]),a,b,g,h,q,r)

p = 0
import time


"""
stime = time.time()
compval = comparison_integrator(rho_x)
comptime = time.time() - stime


stime = time.time()
monval = monte_carlo(rho_x, limits_x, constraint_x, N = 100000)
montime = time.time() - stime


print(f"Exact: {compval[0]:0.4f} error {compval[1]:0.2e} time: {comptime*1000:0.0f}ms")
print(f"Monte-Carlo: {monval:0.4f}  time: {montime*1000:0.0f}ms")
"""

ps = np.linspace(0,10,40)
Ns = 10**np.linspace(1,5,30)
Ns = np.int32(Ns)

realvals = np.zeros(len(ps))
montevals = np.zeros((len(Ns),len(ps)))

for i in tqdm(range(len(ps))):
    p = ps[i]

    J = lambda x: rho_x(x) * x[0]**2

    compval = comparison_integrator(J)
    realvals[i] = compval[0]

    for j in tqdm(range(len(Ns)), leave = False):
        monval = monte_carlo(J, limits_x, constraint_x, N = Ns[j])
        montevals[j,i] = monval

fig, ax = plt.subplots(1,2)
ax, ax2 = ax

ax2.plot(ps,realvals, color = "black", linestyle = "dashed", label = "Točna rešitev")
ax2.set_title("Točna rešitev")
ax2.set_xlabel("p")
ax2.set_ylabel("$J$")
ax2.set_ylim(0,np.max(montevals))

cmap = cm.get_cmap("plasma")
norm = colors.LogNorm(vmin=np.min(Ns), vmax=np.max(Ns))

for i in range(len(Ns)):
    ax.plot(ps,montevals[i,:], color = cmap(norm(Ns[i])))


ax.set_title("Monte-Carlo Rešitve")
ax.set_xlabel("p")
ax.set_ylabel("J")

fig.colorbar(cm.ScalarMappable(norm,cmap), ax = ax, label = "N")

plt.show()
        

