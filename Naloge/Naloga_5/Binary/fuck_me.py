import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import lambertw

def a(t, a0, alpha):
    print(lambertw(1/alpha * np.exp(t/alpha + 1/(alpha * a0) - 1)))
    print(1/alpha * lambertw(1/alpha * np.exp(t/alpha + 1/(alpha * a0) - 1)))

    return 1/alpha * 1/lambertw(1/alpha * np.exp(t/alpha + 1/(alpha * a0) - 1))

ts = np.linspace(0,10000,100)
plt.plot(ts, [a(t, 3, 3) for t in ts])
plt.show()