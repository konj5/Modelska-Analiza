import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm, colors
from tqdm import tqdm
import re

from scipy.sparse.csgraph import shortest_path



def length(r, sij):
    #print(r)
    #print(r[0,:])
    return np.sum(sij[r[1:], r[:-1]])


def procedure(r0, sij, Nmin):
    r0 = r0.T
    L0 = length(r0, sij)
    
    Lmin = L0
    rmin = r0
    for _ in tqdm(range(Nmin)):
        """if _ in (np.int32(np.linspace(0,Nmin,5))):
                plt.figure(figsize=(4,4))
                plt.grid()
                for i in range(len(rmin[0,:])-1):
                    print((rmin[0,i], rmin[1,i]))
                    plt.arrow(rmin[0,i], rmin[1,i], rmin[0,i+1]-rmin[0,i], rmin[1,i+1]-rmin[1,i], head_width = 0.03, head_length=0.02, length_includes_head = True)
                plt.scatter(rmin[0,:],rmin[1,:])
                plt.title(f"korak {_}, dolžina = {Lmin:0.2f}")
                

                print("\n")
                print(rmin[:,:])
                print("\n")

                plt.savefig(f"1.1 step={_}.png")
                
                plt.show()"""

        while(True):
            i = np.random.randint(0,len(r0))
            j = np.random.randint(0,len(r0))
            while i == j:
                j = np.random.randint(0,len(r0))

            r = np.copy(r0); r[i], r[j] = r0[j], r0[i]
            L = length(r, sij)


            

            if L < Lmin:
                Lmin = L
                rmin = r

            if np.random.random() < np.exp((L0-L)/T):
                r0 = r
                L0 = L
                break

    return rmin, Lmin

"""T = 0.1
np.random.seed(1)
r0 = np.random.random((15,2))

Nmin = 50000
r, L = procedure(r0, Nmin)

plt.figure(figsize=(4,4))
plt.grid()

for i in range(len(r[0,:])-1):
    plt.arrow(r[0,i], r[1,i], r[0,i+1]-r[0,i], r[1,i+1]-r[1,i], head_width = 0.03, head_length=0.02, length_includes_head = True)

plt.scatter(r0.T[0,:],r0.T[1,:])
plt.title(f"korak {Nmin}, dolžina = {L:0.2f}")

plt.savefig(f"1.1 step={Nmin}.png")

plt.show()"""

"""mat = np.array([
    [0,1,0,0],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0]
    ])

print(shortest_path(mat))"""

T = 1

points = [[0,0], [0,1.5], [0,2], [0.5,1.5], [1,2], [3,0], [3,2],[3.5,1.5], [4,0], [4,2], [6,0], [6,1.5], [6,2], [6.5,1.5], [7,2]]
connections = {1:(2,4), 2:(1,4,3), 3:(2,5), 4:(1,2,5), 5:(4,7), 6:(4,7,9), 7:(5,6,8), 8:(7,10), 9:(6,10), 10:(8,13,11), 11:(10,12), 12:(11,13,14), 13:(10,12,15), 14:(12,15), 15:(13,14)}
thick = {1:[2], 2:[1,4,3], 3:[2,5], 4:[2], 5:[3], 6:[7], 7:[6,8], 8:[7,10], 9:[10], 10:[8,9], 11:[12], 12:[14,11,13], 13:[15,12], 14:[12], 15:[13]}
arpoints = np.array(points)

mat = np.zeros((len(points), len(points)))

for i in range(len(points)):
    for j in range(len(points)):
        if j+1 in connections[i+1]:
            mat[i,j] = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)

print(mat)
sij = shortest_path(mat)
print(sij)

Nmin = 100000
#r0 = np.array([i for i in range(len(points))])
r0 = np.random.permutation(len(points))
rmin, Lmin = procedure(r0, sij, Nmin)

print(rmin+1)



plt.scatter(arpoints[:,0], arpoints[:,1], c = "black", s = 200)

for key in connections.keys():
    plt.annotate(f"{key}",(points[key-1][0]-0.14, points[key-1][1]-0.02), color = "white")
    for other in connections[key]:
        width = 1
        if other in thick[key]:
            width = 3

        plt.plot([points[key-1][0], points[other-1][0]], [points[key-1][1], points[other-1][1]], color = "black", linewidth=width)

plt.title(f"L = {Lmin:0.2f}")
plt.show()
