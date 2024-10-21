import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
import time
from tqdm import tqdm

#DEFINE FUNCTION:
datas = []
costs = []
runtimess = []
bests = []

w0 = 1.2
wmax = None

#Ns = np.arange(3,100,(100-2)//6)
Ns = np.arange(3,100,(100-2)//6)
for r in tqdm(range(len(Ns))):
    N = Ns[r]

    def F(w, args):
            λ, w0, wmax = args
            global N

            temp = np.zeros(len(w) + 1); temp[0] = w0; temp[1:] = w; w = temp

            F1 = 0
            for i in range(1,N):
                F1 += ((w[i]-w[i-1])*N)**2

            for i in range(0,N):
                F1 -= λ * w[i]
            
            if wmax is not None:
                F2 = F1
                for i in range(1,N-1):
                    F2 += np.exp(w[i] - wmax)

                return F2
            
            return F1
    
                
    #FIND OPTIMAL x:
    #methods = ["Nelder-Mead", "Powell", "CG" ,"BFGS"]
    methods = ["Nelder-Mead","BFGS"]
    N_tries = 100

    data = np.zeros((len(methods), N_tries, N))
    cost = np.zeros((len(methods), N_tries))
    runtimes = np.zeros((len(methods), N_tries))


    for i, method in enumerate(methods):
        methtime = 0
        startstates = []
        for __ in range(N_tries):
            if __ == 0: w0s = np.zeros(N)
            if __ == 1: w0s = np.ones(N)
            w0s = np.random.random(N) * 2
            startstates.append((w0s))

        for j in tqdm(range(len(startstates)), desc = method, leave=False):
            x0 =  startstates[j]

            def getMinimizedSolution(λ, w0, wmax):
                global x0
                result = minimize(F, x0, method = method, args=[λ, w0, wmax])
                return result.x
            
            def findCorrectSolution(w0, wmax):
    
                def f(λ):
                    ws = getMinimizedSolution(λ, w0, wmax)
                    #print(f"λ: {λ},   difference: {1 - np.sum(ws)/N}")
                    return 1 - np.sum(ws)/N

                result = root_scalar(f, x0=0, xtol = 0.01)
                λ = result.root

                return getMinimizedSolution(λ, w0, wmax)

            """
            if r >= 2 and (method == "Powell" or method == "CG"):
                cost[i,j:] = np.max(cost[:,:]) + 1
                break
            """

                
            try:
                stime = time.time()
                ws = findCorrectSolution(w0,None)
                etime = time.time()
                runtime = etime - stime
            except:
                etime = time.time()
                runtime = etime - stime
                data[i,j,:] = ws
                cost[i,j:] = np.max(cost[i,:j+1]) + 1
                runtimes[i,j:] = np.average(runtimes[i,:j+1])
                break
            
            
            methtime += runtime
            if methtime >= 0.5*60 and j > 30:
                cost[i,j:] = np.max(cost[i,:j+1]) + 1
                runtimes[i,j:] = np.average(runtimes[i,:j+1])
                break
            
            


            data[i,j,:] = ws
            cost[i,j] = F(ws, [0, w0, wmax])
            runtimes[i,j] = runtime


    #Process data into shape

    best_x = np.zeros((len(methods), N))
    best_c = np.zeros((len(methods)))
    for i, method in enumerate(methods):
        j = np.argmin(cost[i,:])
        best_x[i,:] = data[i, j, :]
        best_c[i] = cost[i,j]


    

    bests.append(best_x)
    datas.append(data)
    costs.append(cost)
    runtimess.append(runtimes)

avgtimes = np.zeros((len(Ns), len(methods)))
successes = np.zeros((len(Ns), len(methods)))
for i, N in enumerate(Ns):
    for j, method in enumerate(methods):
        min = np.min(costs[i])
        successes[i,j] = np.sum(costs[i][j,:]/N < min/N +1)
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

def anal_sol(T, w0):
    return w0 + 3*(1-w0)*(T-T**2/2)



for i, sol in enumerate(bests):
    temp = np.zeros(N+1); temp[0] = w0; temp[1:] = sol; sol = temp
    plt.plot(np.linspace(0,1,N+1), sol, label = f"N = {Ns[i]}")

Ts = np.linspace(0,1,300)
plt.plot(Ts, [anal_sol(T, w0) for T in Ts], label = "analitična rešitev")





