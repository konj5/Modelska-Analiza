import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


import phaseportrait

r = 1.5
p = 0.1

def dz(A, F):
    return r - p * A * (F+1), F/p * (A-1)

SimplePendulum = phaseportrait.PhasePortrait2D(dz, [0, 16], MeshDim=40, Title=f'p={p:0.2f}, r={r:0.2f}', xlabel=r"$z$", ylabel=r"$l$")
SimplePendulum.plot()

plt.scatter([r/p], [0], c = "Red")
if r > p:
    plt.scatter([1], [r/p-1], c = "Red")

plt.savefig(f"phaseportrait_p={p:0.2f}.png")
plt.show()