import numpy as np
from matplotlib import pyplot as plt

xmin = 0
xmax = 5
xdata = np.linspace(xmin,xmax,300, endpoint=False)

fig, axs = plt.subplots(2,1, figsize = (9,3))
ax1, ax2 = axs
fig.tight_layout(rect=[0.01, 0.03, 1, 1])


def scuffed_definite_integral(xdata, ydata):
    ret = []
    for i in range(len(xdata)):
        ret.append(np.trapezoid(y = ydata[:i], x = xdata[:i]))

    return ret

a = 0.1
wn = 1
an = a
for w0 in [0,0.5,1,1.5,2,5]:
    wn = w0

    def f(x):
        n = x // 1
        x = x % 1

        global wn, an
        if x == 0 and n != 0:
            new_wn = -2*wn - 1/2 * an + 3
            new_an = -6*wn - 2 * an + 6

            wn = new_wn
            an = new_an


        return  - (3*an+6*(wn-1))/2 * x**2 + an*x + wn

    ydata = np.array([f(x) for x in xdata])

    ax1.plot(xdata, np.abs(ydata), label = f"$\\omega_0$ = {w0}")

    ydata = scuffed_definite_integral(xdata, ydata)

    ax2.plot(xdata, np.abs(ydata), label = f"$\\omega_0$ = {w0}")

ax1.set_xlabel("$\\tau$")
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylabel("$|\\omega|$")
ax2.legend()
ax2.set_xlabel("$\\tau$")
ax2.set_ylabel("$|y|$")
plt.show()