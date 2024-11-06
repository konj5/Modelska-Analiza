import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


import phaseportrait

for p in [1.5,2,3]:

    def dz(z, l):
        return p * z * (1-l), l/p * (z-1)

    SimplePendulum = phaseportrait.PhasePortrait2D(dz, [0, 5], MeshDim=40, Title=f'p={p:0.2f}', xlabel=r"$z$", ylabel=r"$l$")
    SimplePendulum.plot()

    plt.savefig(f"phaseportrait_p={p:0.2f}.png")
    plt.show()