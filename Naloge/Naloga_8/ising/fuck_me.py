import random
import matplotlib.pyplot as plt
import numpy as np

def pbc(i,velikost):
    if i>=velikost:
        return 0
    if i<0:
        return velikost-1
    else:
        return i

def energija(mreza):
    N=len(mreza)
    vsota=0
    for i in range(len(mreza)):
        for j in range(len(mreza)):
            vsota+=-(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])*mreza[i][j]
    return vsota/(len(mreza))**2


def energija_kvadrat(mreza):
    vsota=0
    N=len(mreza)
    for i in range(N):
        for j in range(len(mreza)):
            vsota+=(-(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])*mreza[i][j])**2
    return vsota/(len(mreza))**2


def kapaciteta(mreza,temperatura):
    return (energija_kvadrat(mreza)-energija(mreza)**2)/(temperatura**2)


def magnetizacija_kvadrat(matrika):
    vsota = 0
    for vrsta in range(len(matrika)):
        for stolpec in range(len(matrika[0])):
            vsota += matrika[vrsta][stolpec]**2
    return vsota / ((len(matrika)) ** 2)


def susceptibilnost(mreza,temperatura):
    return (magnetizacija_kvadrat(mreza)-magnetizacija(mreza)**2)/temperatura




def ena_nic():
    return random.choice([-1,1])

def magnetizacija(matrika):
    vsota = 0
    for vrsta in range(len(matrika)):
        for stolpec in range(len(matrika[0])):
            vsota += matrika[vrsta][stolpec]
    return vsota / ((len(matrika)) ** 2)



def ising(N,temperatura):
    mreza=[ [ 1 for i in range(N)] for j in range(N)]
    mcs=0
    while (mcs<200):
        for i in range(N):

            for j in range(N):
                # print(j,i)
                # print(pbc(j,N),pbc(i,N))
                delta=-1*(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])*mreza[i][j]
                #print(delta)
                if np.random.rand()<np.exp(2*delta/temperatura):
                    mreza[i][j]*=-1
        mcs+=1
    return mreza



t=np.arange(0.1,5,0.05)
m=[]
sus=[]
kapac=[]


ener=[]
energ_kvadrat=[]

from tqdm import tqdm

for i in tqdm(range(len(t))):
    k = t[i]
    a=ising(50,k)
    m.append(abs(magnetizacija(a)))
    sus.append(susceptibilnost(a,k))
    kapac.append(kapaciteta(a,k))
    ener.append(energija(a))
    energ_kvadrat.append(energija_kvadrat(a))
    print(k)
a1=np.array(ener)
a2=np.array(energ_kvadrat)

print(a1**2)
print(a2)

a3=(a2-a1**2)
print(a3)
a3=a3/t**2

plt.plot(t,a1**2)
plt.plot(t,a2)
plt.grid()
plt.show()

plt.plot(t,kapac,'o')
plt.axvline(x=2.269,color='r')
plt.ylabel('$c \\cdot k_b$')
plt.xlabel('k_bT')
plt.grid()
plt.savefig('specificna_0.png')
plt.show()

plt.plot(t,a3)
plt.axvline(x=2.269,color='r')
plt.show()


plt.plot(t,sus,'o')
plt.axvline(x=2.269,color='r')
plt.ylabel('\\chi')
plt.xlabel('k_bT')
plt.grid()
plt.savefig('susceptibilnost_0.png')
plt.show()


plt.plot(t,m,'o')
plt.axvline(x=2.269,color='r')
plt.ylabel('M')
plt.xlabel('k_bT')
plt.grid()
plt.savefig('magnetizacija_0.png')
plt.show()


plt.matshow(ising(15,15,0),  cmap=plt.cm.gray)
plt.colorbar()
plt.show()