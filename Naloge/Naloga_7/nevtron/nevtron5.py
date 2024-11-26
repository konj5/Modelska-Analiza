import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import re

#Izotropni this time

def run_single():
    r = np.array([0,0,0], np.float64)
    dir = np.array([0,0,1], np.float64)
    costheta = 1
    phi = 0

    i = 0
    while(True):
        s = -lp * np.log(np.random.random())
        r += s * dir

        if r[2] < 0:
            return False, [r, phi, np.arccos(costheta)], i
        if r[2] > d:
            return True, [r, phi, np.arccos(costheta)], i
        
        i += 1

        phi = 2*np.pi*np.random.random()
        costheta = 2*np.random.random()-1
        sintheta = np.sqrt(1-costheta**2)

        dir = np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])
        
def run_N(N):
    data = []
    endstates = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T,pos,n = run_single()
        data.append(int(T))
        endstates.append(pos)
        bounces.append(n)
    
    return data, endstates, bounces


alpha = 1/2
beta = 1/2
d = 1

lp = alpha * d
N = 10**6


ls = np.linspace(0.2,2,40)
data = np.zeros((2,len(ls)))
dataT = -np.ones((2,len(ls)))
dataR = -np.ones((2,len(ls)))
for i in tqdm(range(len(ls)), leave = False):
    lp = ls[i]
    TR, endstates, bounces = run_N(N)

    #print(endstates)
    #print(endstates[0])
    #print(endstates[0][0])
    #print(endstates[0][0][0])
    #print(endstates[0][0][1])


    temp = []
    tempT = []
    tempR = []
    for j in range(len(TR)):
        temp.append(np.sqrt((endstates[j][0][0])**2 +(endstates[j][0][1])**2))
        if TR[j] == 1:
            tempT.append(temp[-1])
        else:
            tempT.append(temp[-1])
            
    """
    print(temp)
    print(np.average(temp))

    print(tempT)
    print(np.average(tempT))

    print(tempR)
    print(np.average(tempR))
    """

    data[0,i] = np.average(temp)
    data[1,i] = np.std(temp)

    dataT[0,i] = np.average(tempT)
    dataT[1,i] = np.std(tempT)

    dataR[0,i] = np.average(tempR)
    dataR[1,i] = np.std(tempR)

    
        



plt.plot(ls, data[0,:])
#plt.fill_between(ls, data[0,:] - data[1,:], data[0,:] + data[1,:], label = "Standardna deviacija", color = "blue", alpha = 0.6)

#plt.plot(ls, dataT[0,:], label = "Prepuščeni", color = "green")
#plt.fill_between(ls, dataT[0,:] - dataT[1,:], dataT[0,:] + dataT[1,:], color = "green", alpha = 0.6)

#plt.plot(ls, dataR[0,:], label = "Odbiti", color = "red")
#plt.fill_between(ls, dataR[0,:] - dataR[1,:], dataR[0,:] + dataR[1,:], color = "red", alpha = 0.6)

plt.ylim(bottom=0)
plt.xlabel("$l_p/d$")
plt.ylabel("$\\langle\\frac{|\\vec{r}_{\\text{začetek}} - \\vec{r}_{\\text{konec}}|}{d}\\rangle$")
#plt.legend()
plt.show()
