import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
import time

N = 4
#method = "Nelder-Mead"
#method = "Powell"
method = "BFGS"


def F(w, args):
    λ, w0, wmax = args
    global N

    temp = np.zeros(len(w) + 1); temp[0] = w0; temp[1:] = w; w = temp

    F1 = 0
    for i in range(1,N+1):
        F1 += ((w[i]-w[i-1])*N)**2

    for i in range(0,N):
        F1 -= λ * w[i]
    
    if wmax is not None:
        F2 = F1
        for i in range(1,N-1):
            F2 += np.exp(w[i] - wmax)

        return F2
    
    return F1

def getMinimizedSolution(λ, w0, wmax):
    stime = time.time()
    result = minimize(F, [0 for x in range(N)], method = method, args=[λ, w0, wmax])
    etime = time.time()
    runtime = etime - stime
    return result.x


def findCorrectSolution(w0, wmax):
    
    def f(λ):
        ws = getMinimizedSolution(λ, w0, wmax)
        print(f"λ: {λ},   difference: {1 - np.sum(ws)/N}")
        return 1 - np.sum(ws)/N

    result = root_scalar(f, x0=0, xtol = 0.01)
    λ = result.root

    return getMinimizedSolution(λ, w0, wmax)

w0 = 0.2

Ns = np.arange(3,100, (100-3)//5)
print(Ns)

def anal_sol(T, w0):
    return w0 + 3*(1-w0)*(T-T**2/2)

w0 = 0.2
wmax = None

fig, axs = plt.subplots(2,1)

for i, N in enumerate(Ns):
    print(f"i:{i}    N:{N}")
    ws = findCorrectSolution(w0,None)
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


        
    