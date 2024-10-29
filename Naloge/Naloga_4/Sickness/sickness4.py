import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import cm, colors
from tqdm import tqdm

def f(t,v, args):
    vdot = np.zeros_like(v)
    alpha, beta, gamma, delta = args

    for i in range(12):
        vdot[4*i+0] = -alpha * v[4*i+0] * v[4*i+1]
        vdot[4*i+1] = alpha * v[4*i+0] * v[4*i+1] - beta * v[4*i+1] - gamma * v[4*i+1]
        vdot[4*i+2] = beta * v[4*i+1]
        vdot[4*i+3] = gamma * v[4*i+1]

    
    connections = [[4,1], [1,2], [1,5], [1,6], [1,7], [1,3], [11,6], [11,5], [5,4], [6,7],[2,7],[7,8], [2,8], [3,8], [2,3], [3,9], [3,10], [9,10], [10,12]]
    
    for connection in connections:
        x,y = connection; x = x-1; y = y-1

        vdot[4*x + 0] += -delta * v[4*x + 0] * v[4*y + 1]
        vdot[4*x + 1] += delta * v[4*x + 0] * v[4*y + 1]
        vdot[4*y + 0] += -delta * v[4*y + 0] * v[4*x + 1]
        vdot[4*y + 1] += delta * v[4*y + 0] * v[4*x + 1]

    

    print(t)

    return vdot


def solve(v0,f,args,t0, tk):
    sol = solve_ivp(lambda t, v: f(t, v, args), [t0,tk], v0, max_step = (tk-t0)/100, atol = 1000)
    return sol.t, sol.y


v0 = np.zeros(4 * 12)
populations = [556862, 36942, 259306, 210747, 118202, 53400, 146429, 75749, 70648, 327858, 118426, 114163]
names = ["Osrednjeslovenska", "Zasavska", "Savinjska", "Gorenjska", "Goriška", "Primorsko-Notranjska", "Jugovshodna Slovenija", "Posavska", "Koroška", "Podravska", "Obalno-Kraška", "Prekmurska"]
v0[0::4] = np.array(populations) / np.sum(populations)
#v0[0::4] = np.array(populations)
v0[1] =  0.1

alpha = 1
beta = 0.07
gamma = 0.01
delta = 0.5

ts, vs = solve(v0, f, [alpha, beta, gamma, delta], 0, 100)

vs *= np.sum(populations)

fig, axs = plt.subplots(6,2)

k = 0
for j in range(6):
    for i in range(2):
        Z, B, O, M = vs[4*k + 0, :], vs[4*k + 1, :], vs[4*k + 2, :], vs[4*k + 3, :]
    

        axs[j,i].plot(ts, Z, label = "Zdravi")
        axs[j,i].plot(ts, B, label = "Bolni")
        axs[j,i].plot(ts, O, label = "Imuni")
        axs[j,i].plot(ts, M, label = "Mrtvi")
        axs[j,i].set_title(names[k])

        k += 1

axs[-1,-1].legend()


plt.show()
        






