import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from tqdm import tqdm

#DEFINE FUNCTION:
datas = []
costs = []
runtimess = []
Ns = np.arange(2,10,1)
for r in tqdm(range(len(Ns))):
    N = Ns[r]

    es = [1 for i in range(N)]

    def F(x):
        assert len(x)%2 == 0
        N = len(x)//2 + 1

        thetas = x[0::2]
        phis = x[1::2]

        temp = np.zeros(len(thetas)+1); temp[0] = 0; temp[1:] = thetas; thetas = temp
        temp = np.zeros(len(phis)+1); temp[0] = 0; temp[1:] = phis; phis = temp

        val = 0
        for i in range(N):
            for j in range(i):
                val += es[i] * es[j] / np.sqrt(
                    (np.sin(thetas[i])*np.cos(phis[i]) - np.sin(thetas[j])*np.cos(phis[j]))**2 + 
                    (np.sin(thetas[i])*np.sin(phis[i]) - np.sin(thetas[j])*np.sin(phis[j]))**2 + 
                    (np.cos(thetas[i]) - np.cos(thetas[j]))**2
                    )
                
        return val
                
    #FIND OPTIMAL x:
    methods = ["Nelder-Mead", "Powell", "CG" ,"BFGS"]
    N_tries = 100

    data = np.zeros((len(methods), N_tries, 2*(N-1)))
    cost = np.zeros((len(methods), N_tries))
    runtimes = np.zeros((len(methods), N_tries))


    for i, method in enumerate(methods):
        startstates = []
        for __ in range(N_tries):
            thetas0 = np.array([np.random.random()*np.pi for _ in range(N)])
            phis0 = np.array([np.random.random()*2*np.pi for _ in range(N)])
            startstates.append((thetas0, phis0))

        for j in tqdm(range(len(startstates)), desc = method, leave=False):
            thetas0, phis0 = startstates[j]

            x0 = np.zeros(2*(N-1))
            for k in range(N-1):
                x0[2*k] = thetas0[k]
                x0[2*k+1] = phis0[k]

            stime = time.time()
            result = minimize(F, x0, method = method)
            etime = time.time()
            runtime = etime - stime

            data[i,j,:] = result.x
            cost[i,j] = F(result.x)
            runtimes[i,j] = runtime


    #Process data into shape

    best_x = np.zeros((len(methods), 2*(N-1)))
    best_c = np.zeros((len(methods)))
    for i, method in enumerate(methods):
        j = np.argmin(cost[i,:])
        best_x[i,:] = data[i, j, :]
        best_c[i] = cost[i,j]

    
    """
    #Plot scatter

    for i, method in enumerate(methods):
        plt.scatter(np.arange(i*N_tries, (i+1)*N_tries, 1),cost[i,:]/N, label = method)

    plt.legend()
    plt.ylabel("Brezdimenzijska energija")
    plt.xlabel("Zaporedno število poskusa")
    plt.show()
    """

    datas.append(data)
    costs.append(cost)
    runtimess.append(runtimes)

avgtimes = np.zeros((len(Ns), len(methods)))
successes = np.zeros((len(Ns), len(methods)))
for i, N in enumerate(Ns):
    for j, method in enumerate(methods):
        min = np.min(costs[i])
        successes[i,j] = np.sum(costs[i][j,:]/N < min/N +0.001)
        avgtimes[i,j] = np.average(runtimess[i][j,:])

success_rate = successes / N_tries


fig, axs = plt.subplots(2,1)
ax1, ax2 = axs

for j, method in enumerate(methods):
    ax1.plot(Ns, success_rate[:,j], label = method, marker = "o", linestyle='--')

    ax2.plot(Ns, avgtimes[:,j], label = method, marker = "o", linestyle='--')

ax1.set_xlabel("Število nabojev")
ax2.set_xlabel("Število nabojev")

ax1.set_ylabel("Delež uspešnih poskusov")
ax2.set_ylabel("Čas računanja[s]")
ax2.set_yscale("log")

ax1.legend()

plt.show()




