import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re


alpha = 1/2
beta = 1/2
d = 1

lp = alpha * d



def run_single():
    x = 0
    dir = 1

    i = 0
    while(True):
        s = -lp * np.log(np.random.random())
        x += dir * s
        i += 1
        if x < 0:
            return False, i
        if x > d:
            return True, i
        
        dir = 1 if np.random.random() >= 0.5 else -1
        
def run_N(N):
    data = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T,n = run_single()
        data.append(int(T))
        bounces.append(n)
    
    return data, bounces

def run_N(N, lp_par):
    global lp
    lp = lp_par
    
    data = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T,n = run_single()
        data.append(int(T))
        bounces.append(n)
    
    return data, bounces

if __name__ == "main":
    #### ERROR

    Ntries = 100
    Ns = np.int32(10**np.linspace(1,6,40)[::-1])

    STDs = np.zeros(len(Ns))
    points = np.zeros((len(Ns), Ntries))
    times = np.zeros(len(Ns))

    import time
    for i in tqdm(range(len(Ns))):
        N = Ns[i]
        stime = time.time()
        for j in tqdm(range(Ntries), leave=False):
            data, bounces = run_N(N)
            points[i,j] = np.sum(data)/N
        times[i] = (time.time()-stime)/Ntries

        STDs[i] = np.std(points[i,:])

    from scipy.optimize import curve_fit
    def f(x,a,b): return a / np.sqrt(x-b)
    params, *_ = curve_fit(f, Ns, STDs, p0=[1,0])
    a,b = params

    fig, axs = plt.subplots(1,3)
    ax1,ax2,ax3 = axs

    for i in range(len(Ns)):
        ax1.scatter(Ntries * [Ns[i]], points[i,:], color = "gray", s = 3)

    ax1.set_xlabel("N")
    ax1.set_ylabel("Delež prepuščenih nevtronov")
    ax1.set_xscale("log")

    ax2.scatter(Ns,STDs, label = "Standardne deviacije")
    ax2.plot(10**np.linspace(1,5,1000), f(10**np.linspace(1,5,1000), a,b), label="$\\frac{a}{\sqrt{N-b}}$", linestyle = "dashed", c = "red")
    ax2.set_yscale("log")
    ax2.set_xscale("log")

    ax2.set_xlabel("N")
    ax2.set_ylabel("Standardna deviacija")
    ax2.legend()

    ax3.plot(Ns,times)
    #ax3.set_yscale("log")
    #ax3.set_xscale("log")

    ax3.set_xlabel("N")
    ax3.set_ylabel("Čas računanja[s]")

    plt.show()

