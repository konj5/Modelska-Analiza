import numpy as np
from matplotlib import pyplot as plt

N = 5
Ls = [1,1,1,3,2]
ts = [1,3,1,2,3]

xmin = 0
xmax = np.sum(ts)
xdata = np.linspace(xmin,xmax,300, endpoint=False)

fig, axs = plt.subplots(2,1, figsize = (9,3))
ax1, ax2 = axs
fig.tight_layout(rect=[0.01, 0.03, 1, 1])


def scuffed_definite_integral(xdata, ydata):
    ret = []
    for i in range(len(xdata)):
        ret.append(np.trapezoid(y = ydata[:i], x = xdata[:i]))

    return ret

def alpha(n):
    global Ls, ts
    return  Ls[n]/Ls[n-1]

def beta(n):
    global Ls, ts
    return (ts[n-1]/ts[n]) * (Ls[n]/Ls[n-1])


for v0 in [0,0.5,1,1.5,2]:

    A,B = None, None

    def f(x):
        global A,B

        if A is None:
            if N == 1:
                M = np.zeros((2*N,2*N))
                b = np.zeros((2*N))
                M[:,0] = [1,1/4]
                M[:,1] = [0,1]
            
            else:
                M = np.zeros((2*N,2*N))
                b = np.zeros((2*N))

                M[0:3,0] = np.array([0,1,0]); b[0] = v0 * ts[-1]/Ls[-1]
                for i in range(1,2*N-1,2):
                    M[i-1:i+2,i] = np.array([2,6,1 * alpha(i//2)]); b[i] = 6
                    M[i-1:i+3,i+1] = np.array([1,4,0,2 * beta(i//2)]); b[i+1] = 6
                M[-3:,-1] = np.array([0,5,12]); b[-1] = 12

            M = M.T
            coefs = np.linalg.solve(M, b)
            A = coefs[0::2]
            B = coefs[1::2]

        #print(f"{x:0.3f},  {n:0.3f},  {x % 1:0.3f},  {A[n]:0.3f},  {B[n]:0.3f}, {-1/4 * (6 * A[n] + 12*B[n] - 12)*(x%1)**2 + A[n]*(x%1) + B[n]:0.3f}")

        T = 0
        n = 0
        for i in range(len(ts)):
            T += ts[i]
            if T >= x:
                n = i
                break

        x = x - (T - ts[n])


        return ( -1/4 * (6 * A[n] + 12*B[n] - 12)*(x/ts[n])**2 + A[n]*(x/ts[n]) + B[n]) * Ls[n-1] / ts[n-1]

    ydata = np.array([f(x) for x in xdata])

    ax1.plot(xdata, ydata, label = f"$v_0$ = {v0}")

    ydata = scuffed_definite_integral(xdata, ydata)

    ax2.plot(xdata, ydata, label = f"$v_0$ = {v0}")

T = 0
ax1.axvline(T, 0, 2, color = "black", linestyle = "dashed", label = "semafor")
for t in ts:
    T += t
    ax1.axvline(T, 0, 2, color = "black", linestyle = "dashed")

L = 0
ax2.axvline(L, 0, 2, color = "black", linestyle = "dashed", label = "semafor")
for t in Ls:
    L += t
    ax2.axvline(L, 0, 2, color = "black", linestyle = "dashed")

ax1.set_xlabel("$t$")
ax1.set_ylabel("$v$")
ax2.legend()
ax2.set_xlabel("$t$")
ax2.set_ylabel("$x$")
plt.show()