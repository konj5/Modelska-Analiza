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

def energija(mreza,zunanje):
    N=len(mreza)
    vsota=0
    for i in range(len(mreza)):
        for j in range(len(mreza)):
            vsota+=(-(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])-zunanje)*mreza[i][j]
    return vsota/(len(mreza))**2


def energija_kvadrat(mreza,zunanje):
    vsota=0
    N=len(mreza)
    for i in range(N):
        for j in range(len(mreza)):
            vsota+=((-(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])-zunanje)*mreza[i][j])**2
    return vsota/(len(mreza))**2


def kapaciteta(mreza,temperatura,zunanje):
    return (energija_kvadrat(mreza,zunanje)-energija(mreza,zunanje)**2)/(temperatura**2)


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



def ising(N,temperatura,zunanje_polje):
    mreza=[ [ena_nic() for i in range(N)] for j in range(N)]
    mcs=0
    nakljucna=np.random.random(N**2*200)
    while (mcs<200):
        for i in range(N):

            for j in range(N):
                # print(j,i)
                # print(pbc(j,N),pbc(i,N))
                delta=(-1*(mreza[i][pbc(j+1,N)]+mreza[i][pbc(j-1,N)]+mreza[pbc(i+1,N)][j]+mreza[pbc(i-1,N)][j])-zunanje_polje)*mreza[i][j]
                #print(delta)
                if nakljucna[i+j+mcs]<np.exp(2*delta/temperatura):
                    mreza[i][j]*=-1
        mcs+=1
    return mreza



t=np.arange(0.1,10,0.01)
m=[]
sus=[]
kapac=[]


ener=[]
energ_kvadrat=[]

e0=[]
e1=[]
e5=[]
e10=[]

s0=[]
s1=[]
s5=[]
s10=[]

c0=[]
c1=[]
c5=[]
c10=[]

m0=[]
m1=[]
m5=[]
m10=[]



for k in t:
    h=0.0

    a=ising(50,k,h)
    e0.append(energija(a,h))
    m0.append(abs(magnetizacija(a)))
    s0.append(susceptibilnost(a,k))
    c0.append(kapaciteta(a,k,h))
    print(k,'\t',h)
print(h,'------------------------------0')
for k in t:
    h=0.1

    a=ising(50,k,h)
    e1.append(energija(a,h))
    m1.append(abs(magnetizacija(a)))
    s1.append(susceptibilnost(a,k))
    c1.append(kapaciteta(a,k,h))
    print(k, '\t', h)
print(h,'------------------------------1')
for k in t:
    h=0.5

    a=ising(50,k,h)
    e5.append(energija(a,h))
    m5.append(abs(magnetizacija(a)))
    s5.append(susceptibilnost(a,k))
    c5.append(kapaciteta(a,k,h))
    print(k, '\t', h)
print(h,'------------------------------2')

for k in t:
    h=1

    a=ising(50,k,h)
    e10.append(energija(a,h))
    m10.append(abs(magnetizacija(a)))
    s10.append(susceptibilnost(a,k))
    c10.append(kapaciteta(a,k,h))
    print(k, '\t', h)
print(h,'------------------------------3')
# plt.matshow(a,cmap=plt.cm.gray)
# plt.show()


f=open('podatki_energija.txt','w')

for i in range(len(t)):
    f.write(str(t[i]) + '\t' + str(e0[i]) + '\t' + str(e1[i]) + '\t' + str(e5[i]) + '\t' + str(e10[i]) + '\r\n')
f.close()

f=open('podatki_magnetizacija.txt','w')

for i in range(len(t)):
    f.write(str(t[i]) + '\t' + str(m0[i]) + '\t' + str(m1[i]) + '\t' + str(m5[i]) + '\t' + str(m10[i]) + '\r\n')
f.close()

f=open('podatki_specificna.txt','w')
for i in range(len(t)):
    f.write(str(t[i])+'\t'+str(c0[i])+'\t'+str(c1[i])+'\t'+str(c5[i])+'\t'+str(c10[i])+'\r\n')
f.close()


f=open('podatki_susceptibilnost.txt','w',encoding='utf-8')
for i in range(len(t)):
    f.write(str(t[i]) + '\t' + str(s0[i]) + '\t' + str(s1[i]) + '\t' + str(s5[i]) + '\t' + str(s10[i]) + '\r\n')
f.close()


plt.scatter(t,s0,label='H=0')
plt.scatter(t,s1,label='H=0.1')
plt.scatter(t,s5,label='H=0.5')
plt.scatter(t,s10,label='H=1')

plt.legend()
plt.axvline(x=2.269,color='r')
plt.ylabel('energija')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('energija_tudi_zunanje.png')
plt.show()



plt.scatter(t,s0,label='H=0')
plt.scatter(t,s1,label='H=0.1')
plt.scatter(t,s5,label='H=0.5')
plt.scatter(t,s10,label='H=1')

plt.legend()
plt.axvline(x=2.269,color='r')
plt.ylabel('susceptibilnost')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('susceptibilnost_tudi_zunanje.png')
plt.show()


plt.scatter(t,m0,label='H=0')
plt.scatter(t,m1,label='H=0.1')
plt.scatter(t,m5,label='H=0.5')
plt.scatter(t,m10,label='H=1')

plt.legend()
plt.axvline(x=2.269,color='r')
plt.ylabel('magnetizacija')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('magneizacija_tudi_zunanje.png')
plt.show()


plt.scatter(t,c0,label='H=0')
plt.scatter(t,c1,label='H=0.1')
plt.scatter(t,c5,label='H=0.5')
plt.scatter(t,c10,label='H=1')

plt.legend()
plt.axvline(x=2.269,color='r')
plt.ylabel('specifična toplota')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('specificna_tudi_zunanje.png')
plt.show()


# plt.matshow(ising(15,15,0),  cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()