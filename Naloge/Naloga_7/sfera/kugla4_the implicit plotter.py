import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return eval(qVal)


def queryMatplotlib3dEq(qVal, ax):
    x = np.linspace(-6, 6, num=1000)
    y = np.linspace(-6, 6, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    cmap = plt.colormaps.get_cmap('viridis')
    
    if qVal[0] == '-':
        cmap = cmap.reversed()
        
    ax.plot_surface(X, Y, Z, cmap=cmap)


# solve the equation with sympy
x, y, z = sym.symbols('x y z')
eq = sym.Eq(30, x**2+y**2+z**2)

# replace 'sqrt` with 'np.sqrt'
eqs = [v.replace('sqrt', 'np.sqrt') for v in map(str, sym.solve(eq, z))]

# create the figure and axes outside the function, so axes will be reused for each loop of the equation
fig = plt.figure()
ax = plt.axes(projection='3d')

# setting the limits and aspect ensures the sphere doesn't look like an ellipse
ax.set_zlim(-6, 6)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')

for qVal in eqs:
    print(qVal)
    queryMatplotlib3dEq(qVal, ax)

plt.show()