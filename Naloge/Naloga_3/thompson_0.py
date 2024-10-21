import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

#DEFINE FUNCTION:

N = 8
#method = "Nelder-Mead"
#method = "Powell"
method = "BFGS"

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

thetas0 = np.array([np.random.random()*np.pi for i in range(N)])
phis0 = np.array([np.random.random()*2*np.pi for i in range(N)])

#thetas0 = np.array([np.pi/2, np.pi/2, np.pi])
#phis0 = np.array([0, np.pi, 0])

x0 = np.zeros(2*(N-1))
for i in range(N-1):
    x0[2*i] = thetas0[i]
    x0[2*i+1] = phis0[i]

stime = time.time()
result = minimize(F, x0, method = method)
etime = time.time()
runtime = etime - stime

print(runtime)
print(result.success)
print(result.x)

#PLOT RESULT

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', )

# Make data
r = 1
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z, color='linen', alpha=0.2, zorder = 0)


# plot circular curves over the surface
theta = np.linspace(0, 2 * np.pi, 100)
z = np.zeros(100)
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, color='black', alpha=0.25, zorder = 0)
ax.plot(z, x, y, color='black', alpha=0.25, zorder = 0)

## add axis lines
zeros = np.zeros(1000)
line = np.linspace(-1,1,1000)

ax.plot(line, zeros, zeros, color='black', alpha=0.25, zorder = 0)
ax.plot(zeros, line, zeros, color='black', alpha=0.25, zorder = 0)
ax.plot(zeros, zeros, line, color='black', alpha=0.25, zorder = 0)

#Plot the points
thetas = result.x[0::2]
phis = result.x[1::2]

xs,ys,zs = [], [], []
for i in range(N):
    xs.append(np.sin(thetas[i-1] if i != 0 else 0) * np.cos(phis[i-1] if i != 0 else 0))
    ys.append(np.sin(thetas[i-1] if i != 0 else 0) * np.sin(phis[i-1] if i != 0 else 0))
    zs.append(np.cos(thetas[i-1] if i != 0 else 0))

ax.set_box_aspect([1.0, 1.0, 1.0])

#ax.scatter(xs,ys,zs, color = "red", zorder = 1, s = 100)
#ax.plot_trisurf(xs,ys,zs, zorder = 1, alpha = 0.6)
#plt.show()

#############################
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from matplotlib import cm
from matplotlib import animation

plt.style.use('dark_background')

# Data reader from a .csv file
def getData(file):
    lstX = []
    lstY = []
    lstZ = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            lstX.append(row[0])
            lstY.append(row[1])
            lstZ.append(row[2])
    return lstX, lstY, lstZ

# This function gets rid of the triangles at the base of the neck
# It just filters out any triangles which have at least one side longer than toler
def removeBigTriangs(points, inds, toler=35):
    newInds = []
    for ind in inds:
        if ((np.sqrt(np.sum((points[ind[0]]-points[ind[1]])**2, axis=0))<toler) and
            (np.sqrt(np.sum((points[ind[0]]-points[ind[2]])**2, axis=0))<toler) and
            (np.sqrt(np.sum((points[ind[1]]-points[ind[2]])**2, axis=0))<toler)):
            newInds.append(ind)
    return np.array(newInds)

# this calculates the location of each point when it is expanded out to the sphere
def calcSpherePts(points, center):
    kdtree = KDTree(points) # tree of nearest points
    # d is an array of distances, i is array of indices
    d, i = kdtree.query(center, points.shape[0])
    spherePts = np.zeros(points.shape, dtype=float)
    
    radius = np.amax(d)
    for p in range(points.shape[0]):
        spherePts[p] = points[i[p]] *radius /d[p]
    return spherePts, i # points and the indices for where they were in the original lists
    

x,y,z = xs, ys, zs

pts = np.stack((x,y,z), axis=1)

# generating data
spherePts, sphereInd = calcSpherePts(pts, [0,0,0])
hull = ConvexHull(spherePts)
triangInds = hull.simplices # returns the list of indices for each triangle
triangInds = removeBigTriangs(pts[sphereInd], triangInds)

# plotting!
f = 1.05
for i in range(len(pts[:,0])):
    ax.scatter3D(pts[i,0]*f, pts[i,1]*f, pts[i,2]*f, s=100, c='r', alpha=1.0, zorder = 10)
ax.plot_trisurf(pts[sphereInd,0], pts[sphereInd,1], pts[sphereInd,2], triangles=triangInds, alpha=0.6, zorder = 100)
#############################

color_tuple = (1.0, 1.0, 1.0, 0.0)
ax.tick_params(color=color_tuple, labelcolor=color_tuple)

plt.tight_layout()

plt.show()





#### convex hull za vizualizacijo nabojev!
#### RAZLIČNI NABOJI PERHAPS?