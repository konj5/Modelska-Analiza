import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from tqdm import tqdm

#DEFINE FUNCTION:
datas = []
costs = []
runtimess = []
bests = []

#Ns = np.arange(2,10,1)
Ns = np.arange(2,4,1)
for r in tqdm(range(len(Ns))):
    N = Ns[r]

    es = np.array([2,7,1,8,2,8,1,8,2,8,4,5,9])

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
    #methods = ["Nelder-Mead", "Powell", "CG" ,"BFGS"]
    methods = ["BFGS"]
    N_tries = 100

    data = np.zeros((len(methods), N_tries, 2*(N-1)))
    cost = np.zeros((len(methods), N_tries))
    runtimes = np.zeros((len(methods), N_tries))


    for i, method in enumerate(methods):
        startstates = []
        for __ in range(N_tries):
            thetas0 = np.array([np.random.random()*np.pi for _ in range(N)])
            phis0 = np.array([np.random.random()*2*np.pi for _ in range(N)])
            startstates.append((thetas0, phis0))

        for j in tqdm(range(len(startstates)), desc = method, leave=False):
            thetas0, phis0 = startstates[j]

            x0 = np.zeros(2*(N-1))
            for k in range(N-1):
                x0[2*k] = thetas0[k]
                x0[2*k+1] = phis0[k]

            stime = time.time()
            result = minimize(F, x0, method = method)
            etime = time.time()
            runtime = etime - stime

            data[i,j,:] = result.x
            cost[i,j] = F(result.x)
            runtimes[i,j] = runtime


    #Process data into shape

    best_x = np.zeros((len(methods), 2*(N-1)))
    best_c = np.zeros((len(methods)))
    for i, method in enumerate(methods):
        j = np.argmin(cost[i,:])
        best_x[i,:] = data[i, j, :]
        best_c[i] = cost[i,j]

    
    """
    #Plot scatter

    for i, method in enumerate(methods):
        plt.scatter(np.arange(i*N_tries, (i+1)*N_tries, 1),cost[i,:]/N, label = method)

    plt.legend()
    plt.ylabel("Brezdimenzijska energija")
    plt.xlabel("Zaporedno število poskusa")
    plt.show()
    """

    bests.append(best_x)
    datas.append(data)
    costs.append(cost)
    runtimess.append(runtimes)

avgtimes = np.zeros((len(Ns), len(methods)))
successes = np.zeros((len(Ns), len(methods)))
for i, N in enumerate(Ns):
    for j, method in enumerate(methods):
        min = np.min(costs[i])
        successes[i,j] = np.sum(costs[i][j,:]/N < min/N +0.001)
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

for r, N in enumerate(Ns):



    #PLOT RESULT

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', )
    color_tuple = (1.0, 1.0, 1.0, 0.0)
    ax.tick_params(color=color_tuple, labelcolor=color_tuple)

    fig.set_facecolor('white')
    ax.set_facecolor('white') 
    ax.grid(False) 
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='linen', alpha=0.2, zorder = 0)


    # plot circular curves over the surface
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.zeros(100)
    x = np.sin(theta)
    y = np.cos(theta)

    ax.plot(x, y, z, color='black', alpha=0.25, zorder = 0)
    ax.plot(z, x, y, color='black', alpha=0.25, zorder = 0)

    ## add axis lines
    zeros = np.zeros(1000)
    line = np.linspace(-1,1,1000)

    ax.plot(line, zeros, zeros, color='black', alpha=0.25, zorder = 0)
    ax.plot(zeros, line, zeros, color='black', alpha=0.25, zorder = 0)
    ax.plot(zeros, zeros, line, color='black', alpha=0.25, zorder = 0)

    #Plot the points
    thetas = bests[r][0, 0::2]
    phis = bests[r][0, 1::2]

    xs,ys,zs = [], [], []
    for i in range(N):
        xs.append(np.sin(thetas[i-1] if i != 0 else 0) * np.cos(phis[i-1] if i != 0 else 0))
        ys.append(np.sin(thetas[i-1] if i != 0 else 0) * np.sin(phis[i-1] if i != 0 else 0))
        zs.append(np.cos(thetas[i-1] if i != 0 else 0))

    ax.set_box_aspect([1.0, 1.0, 1.0])

    if N == 2:
        ax.scatter(xs,ys,zs, color = "red", zorder = 1, s = 20*es[0:len(xs)])
        plt.show()
        continue

    if N == 3:
        ax.scatter(xs,ys,zs, color = "red", zorder = 1, s = 20*es[0:len(xs)])
        ax.plot_trisurf(xs,ys,zs, zorder = 1, alpha = 0.6)
        plt.show()
        continue

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
        ax.scatter3D(pts[i,0]*f, pts[i,1]*f, pts[i,2]*f, s = 20*es[i], c='r', alpha=1.0, zorder = 10)
    ax.plot_trisurf(pts[sphereInd,0], pts[sphereInd,1], pts[sphereInd,2], triangles=triangInds, alpha=0.6, zorder = 100)
    #############################

    

    plt.tight_layout()

    plt.show()




