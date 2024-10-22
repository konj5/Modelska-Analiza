import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
import time

N = 4
#method = "Nelder-Mead"
#method = "Powell"
method = "BFGS"


def F(w, args):
    λ, w0, wmax, beta = args
    global N

    temp = np.zeros(len(w) + 1); temp[0] = w0; temp[1:] = w; w = temp

    F1 = 0
    for i in range(1,N+1):
        F1 += ((w[i]-w[i-1])*N)**2

    F0 = 0
    for i in range(0,N):
        F0 -= λ * w[i]
    
    if wmax is not None:
        F2 = 0
        for i in range(1,N-1):
            F2 += np.exp(beta * (w[i] - wmax))
    
    return (F1 + F2) + F0

def getMinimizedSolution(λ, w0, wmax, beta):
    stime = time.time()
    result = minimize(F, [wmax for x in range(N)], method = method, args=[λ, w0, wmax, beta])
    etime = time.time()
    runtime = etime - stime
    print(runtime)
    return result.x


def bisection(f, xtol):
    xmax = 100000; xmin = -1000
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


def findCorrectSolution(w0, wmax, beta):
    
    def f(λ):
        ws = getMinimizedSolution(λ, w0, wmax, beta)
        #print(f"λ: {λ},   difference: {1 - np.sum(ws)/N}")
        return 1 - np.sum(ws)/N

    #result = root_scalar(f, x0=1, xtol = 0.01)
    #λ = result.root
    λ = bisection(f, xtol = 0.01)

    return getMinimizedSolution(λ, w0, wmax, beta)


N = 100

betas = np.arange(0,11,2)[::-1]

#betas = [10]
def anal_sol(T, w0):
    return w0 + 3*(1-w0)*(T-T**2/2)

w0 = 1.2
wmax = 1

fig, ax = plt.subplots(1,1)

for i, beta in enumerate(betas):
    print(f"i:{i}    $\\beta$:{beta}")
    ws = findCorrectSolution(w0,wmax,beta)
    temp = np.zeros(len(ws) + 1); temp[0] = w0; temp[1:] = ws; ws = temp
    ax.plot(np.linspace(0,1,N+1), ws, label = f"$\\beta$ = {betas[i]:0.2f}")





ax.legend()
ax.set_xlabel("$\\tau$")
ax.set_ylabel("$\\omega$")

plt.show()


        
    