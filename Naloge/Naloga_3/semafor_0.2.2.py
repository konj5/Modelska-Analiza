import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
import time

N = 4
#method = "Nelder-Mead"
#method = "Powell"
method = "BFGS"

mintimes = []
proctimes = []

def F(w, args):
    w0, gamma = args
    global N

    temp = np.zeros(len(w) + 1); temp[0] = w0; temp[1:] = w; w = temp

    F1 = 0
    for i in range(1,N+1):
        F1 += ((w[i]-w[i-1])*N)**2


    F2 = F1 + 1 + np.exp(gamma*(np.sum(w)-(w[0]+w[-1])/2 - N)) + np.exp(-gamma*(np.sum(w)-(w[0]+w[-1])/2 - N))
    
    return F2


def bisection(f, xtol):
    xmax = 100; xmin = 0
    xmid = (xmax + xmin) / 2

    midval = f(xmid)
    maxval = f(xmax)
    minval = f(xmin)

    while np.abs(midval) > xtol:
        if (midval < 0 and maxval > 0) or (midval > 0 and maxval < 0):
            xmid, xmin = (xmid + xmax) / 2, xmid
            midval, minval = f(xmid), midval
        else:
            xmid, xmax = (xmid + xmin) / 2, xmid
            midval, maxval = f(xmid), midval

        print(f"{xmin}:{minval}  {xmid}:{midval}  {xmax}:{maxval}")

    return xmid


def findCorrectSolution(w0):
    
    def f(gamma):
        ws = getMinimizedSolution(w0, gamma)
        #print(f"λ: {λ},   difference: {1 - np.sum(ws)/N}")
        return 1 - np.sum(ws)/N

    #result = root_scalar(f, x0=1, xtol = 0.01)
    #λ = result.root
    stime = time.time()
    gamma = bisection(f, xtol = 0.01)
    etime = time.time()
    runtime = etime - stime
    proctimes.append(runtime)

    return getMinimizedSolution(w0, gamma)

def getMinimizedSolution(w0, gamma):
    stime = time.time()
    result = minimize(F, [1 for x in range(N)], method = method, args=[w0, gamma])
    etime = time.time()
    runtime = etime - stime
    mintimes.append(runtime)
    return result.x


def anal_sol(T, w0):
    return w0 + 3*(1-w0)*(T-T**2/2)

N = 100

w0 = 0.2
min = 0
max = 20

fig, axs = plt.subplots(2,1)

Ns = np.arange(3,100, (100-3)//5)

for i, N in enumerate(Ns):
    print(f"i:{i}    N:{N}")
    ws = findCorrectSolution(w0)
    temp = np.zeros(len(ws) + 1); temp[0] = w0; temp[1:] = ws; ws = temp
    axs[0].plot(np.linspace(0,1,N+1), ws, label = f"N = {Ns[i]}", linestyle = "dashed")

    Ts = np.linspace(0,1,N+1)
    axs[1].plot(Ts, np.abs(ws-np.array([anal_sol(T, w0) for T in Ts])), label = f"N = {Ns[i]}")

Ts = np.linspace(0,1,300)
axs[0].plot(Ts, [anal_sol(T, w0) for T in Ts], label = "analitična rešitev")




axs[0].legend()
axs[0].set_xlabel("$\\tau$")
axs[0].set_ylabel("$\\omega$")

axs[1].legend()
axs[1].set_xlabel("$\\tau$")
axs[1].set_ylabel("Absolutna napaka")
axs[1].set_yscale("log")

plt.show()


        
    