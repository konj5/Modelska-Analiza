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

Ns = 10**np.linspace(2,3,6)[::-1]
N_attempts = 100
p = 0



realmonvals = []
realNs = []

stdvals = []
calctimes = []

stime = time.time()
compval = comparison_integrator(rho_x)
comptime = time.time() - stime
exact_val = compval[0]

for i in tqdm(range(len(Ns))):
    N = int(Ns[i])
    monvals = []
    montimes = []
    for _ in tqdm(range(N_attempts), leave = False):
        stime = time.time()
        monval = monte_carlo(rho_x, limits_x, constraint_x, N = N)
        montime = time.time() - stime
        montimes.append(montime); monvals.append(monval); realmonvals.append(monval); realNs.append(N)


    STD = np.std(monvals)

    stdvals.append(STD)
    calctimes.append(np.average(montimes))

fig, axs = plt.subplots(1,3)
ax1, ax2, ax3 = axs


ax1.set_title("Izračunana masa")
ax1.scatter(realNs, realmonvals, s = 3, label = "Monte-Carlo rešitev")
ax1.plot(realNs, [exact_val for _ in realNs], linestyle = "dashed", color = "red", label = "Točna rešitev")
ax1.set_xscale("log")
ax1.set_xlabel("N")
ax1.set_ylabel("m")
ax1.legend()

ax2.set_title("Absolutna napaka rešitev")
ax2.plot(Ns, stdvals, label = "Standardna deviacija M-C")
ax2.plot(Ns, [compval[1] for _ in Ns], linestyle = "dashed", color = "red", label = "Napaka scipy.integrate.tplquad")
ax2.legend()
ax2.set_xlabel("N")
ax2.set_ylabel("Absolutna napaka")
ax2.set_xscale("log")
ax2.set_yscale("log")


ax3.plot(Ns, calctimes, label = "Monte-Carlo")
ax3.set_title("Čas računanja")
ax3.plot(Ns, [comptime for _ in Ns], linestyle = "dashed", color = "red", label = "scipy.integrate.tplquad")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("N")
ax3.set_ylabel("čas računanja[s]")
ax3.legend()

plt.show()




