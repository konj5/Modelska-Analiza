import numpy as np
from matplotlib import pyplot as plt

N = 3

xmin = 0
xmax = N
xdata = np.linspace(xmin,xmax,300, endpoint=False)

fig, axs = plt.subplots(2,1, figsize = (9,3))
ax1, ax2 = axs
fig.tight_layout(rect=[0.01, 0.03, 1, 1])


def scuffed_definite_integral(xdata, ydata):
    ret = []
    for i in range(len(xdata)):
        ret.append(np.trapezoid(y = ydata[:i], x = xdata[:i]))

    return ret


for w0 in [0,0.5,1,1.5,2]:

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

                M[0:3,0] = np.array([0,1,0]); b[0] = w0
                for i in range(1,2*N-1,2):
                    M[i-1:i+2,i] = np.array([2,6,1]); b[i] = 6
                    M[i-1:i+3,i+1] = np.array([1,4,0,2]); b[i+1] = 6
                M[-3:,-1] = np.array([0,5,12]); b[-1] = 12

            M = M.T
            print(M)
            print(1/0)

            coefs = np.linalg.solve(M, b)
            A = coefs[0::2]
            B = coefs[1::2]

        n = int(np.floor(x))
        #print(f"{x:0.3f},  {n:0.3f},  {x % 1:0.3f},  {A[n]:0.3f},  {B[n]:0.3f}, {-1/4 * (6 * A[n] + 12*B[n] - 12)*(x%1)**2 + A[n]*(x%1) + B[n]:0.3f}")
        x = x % 1

        return  -1/4 * (6 * A[n] + 12*B[n] - 12)*x**2 + A[n]*x + B[n]

    ydata = np.array([f(x) for x in xdata])

    ax1.plot(xdata, ydata, label = f"$\\omega_0$ = {w0}")

    ydata = scuffed_definite_integral(xdata, ydata)

    ax2.plot(xdata, ydata, label = f"$\\omega_0$ = {w0}")

    wmin, wmax = 0, 5
    while wmax-wmin >= 10**-10:
        wmid = (wmin+wmax)/2
        #print(f"{wmin:0.3f} {wmax:0.3f} {(wmin+wmax)/2:0.3f}")
        w0 = wmid
        A,B = None, None
        ydata = np.array([f(x) for x in xdata])
        for y in ydata:
            if y < 0:
                wmax = wmid
                break
        if wmax != wmid:
            wmin = wmid

    #input()
            
    #print(f"{wmin:0.3f} {wmax:0.3f} {(wmin+wmax)/2:0.3f}")
    A,B = None, None


ax1.set_xlabel("$\\tau$")
ax1.set_ylabel("$\\omega$")
ax2.legend()
ax2.set_xlabel("$\\tau$")
ax2.set_ylabel("$y$")
plt.show()