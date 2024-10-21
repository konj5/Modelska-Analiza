import numpy as np
from matplotlib import pyplot as plt

xmin = 0
xmax = 1
xdata = np.linspace(xmin,xmax,300)

a = -10
b = -a

fig, axs = plt.subplots(1,2, figsize = (9,3))
ax1, ax2 = axs
fig.tight_layout(rect=[0.01, 0.03, 1, 1])


def scuffed_definite_integral(xdata, ydata):
    ret = []
    for i in range(len(xdata)):
        ret.append(np.trapezoid(y = ydata[:i], x = xdata[:i]))

    return ret

for w0 in [0,0.5,1,1.5,2,5]:

    def f(x):
        s = np.sin(b)
        c = np.cos(b)
        t = np.tan(b)

        g = (w0 * (s + t*(1-c)) - b) / (b - (s + t * (1-c)))

        A = w0 + g
        B = A * t

        return A * np.cos(a*x) + B * np.sin(-a*x) - g

    ydata = np.array([f(x) for x in xdata])

    ax1.plot(xdata, ydata, label = f"$\\omega_0$ = {w0}")

    ydata = scuffed_definite_integral(xdata, ydata)

    ax2.plot(xdata, ydata, label = f"$\\omega_0$ = {w0}")

ax1.set_xlabel("$\\tau$")
ax1.set_ylabel("$\\omega$")
ax2.legend()
ax2.set_xlabel("$\\tau$")
ax2.set_ylabel("$y$")
plt.show()