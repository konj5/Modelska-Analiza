import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

R = 1

def lottery_box():
    return [2*np.random.random()-1, 2*np.random.random()-1, 2*np.random.random()-1]

def lottery_path(lp):
    return [2*np.random.random()-1, 2*np.pi*np.random.random(), -lp*np.log(np.random.random())]


def volume_function_sphere(x,y,z):
    if x**2 + y**2 + z**2 < R**2:return True
    return False

def volume_function_star(x,y,z):
    #if np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z))<=1:
        #print(np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z))<=1)
    if np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1:return True
    return False

def volume_function_elipsoid(x,y,z):
    if (2*x)**2 + (3*y)**2 + z**2 < R**2:return True
    return False

def volume_function_cylinder(x,y,z):
    if (x)**2 + (y)**2 < R**2 and z > -R and z < R:return True
    return False

def volume_function_box(x,y,z):
    if x > -R and x < R and y > -R and y < R and z > -R and z < R:return True
    return False


def get_random_hexet(lp):
    while True:
        x,y,z = lottery_box()
        if volume_function(x,y,z):
            break
    costheta, phi, s = lottery_path(lp)

    return x,y,z,costheta,phi,s
        

def run_one_particle(lp):
    x,y,z,costheta,phi,s = get_random_hexet(lp)
    sintheta = np.sqrt(1-costheta**2)

    X,Y,Z = np.array([x,y,z]) + s*np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])

    return int(not volume_function(X,Y,Z))

def run_for_N(N,lp):
    escapees = 0
    for _ in tqdm(range(N), leave=False):
        escapees += run_one_particle(lp)

    return escapees/N
    
volume_functions = [volume_function_sphere, volume_function_cylinder, volume_function_star, volume_function_box, volume_function_elipsoid]
names = ["krogla", "valj", "zvezda", "kocka", "elipsoid"]


ls = np.linspace(0,6,40)
data = np.ones((len(volume_functions), len(ls)))

N = int(10**4.5)

for i in tqdm(range(len(volume_functions))):
    volume_function = volume_functions[i]
    for j in tqdm(range(len(ls)), leave = False):
        lp = ls[j]
        data[i,j] = run_for_N(N, lp)

for i in range(len(volume_functions)):
    plt.plot(ls, data[i,:], label = names[i])

plt.legend()
plt.xlabel("$l_p$")
plt.ylabel("Verjetnost pobega")

plt.show()



