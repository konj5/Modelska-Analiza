import numpy as np
import matplotlib.pyplot as plt

t=[]
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



f=open('podatki_energija.txt','r')
vrstice=f.readlines()
for vrsta in vrstice:
    podatki=(vrsta.strip('\r\n')).split('\t')
    # print(vrsta)
    if podatki!=['']:
        print(podatki)

        t.append(float(podatki[0]))
        e0.append(float(podatki[1]))
        e1.append(float(podatki[2]))
        e5.append(float(podatki[3]))
        e10.append(float(podatki[4]))

f.close()


f=open('podatki_magnetizacija.txt','r')
vrstice=f.readlines()
for vrsta in vrstice:
    podatki=(vrsta.strip('\r\n')).split('\t')
    # print(vrsta)
    if podatki!=['']:
        print(podatki)

        m0.append(float(podatki[1]))
        m1.append(float(podatki[2]))
        m5.append(float(podatki[3]))
        m10.append(float(podatki[4]))

f.close()



f=open('podatki_specificna.txt','r')
vrstice=f.readlines()
for vrsta in vrstice:
    podatki=(vrsta.strip('\r\n')).split('\t')
    # print(vrsta)
    if podatki!=['']:
        print(podatki)


        c0.append(float(podatki[1]))
        c1.append(float(podatki[2]))
        c5.append(float(podatki[3]))
        c10.append(float(podatki[4]))

f.close()


f=open('podatki_susceptibilnost.txt','r')
vrstice=f.readlines()
for vrsta in vrstice:
    podatki=(vrsta.strip('\r\n')).split('\t')
    # print(vrsta)
    if podatki!=['']:
        print(podatki)

        s0.append(float(podatki[1]))
        s1.append(float(podatki[2]))
        s5.append(float(podatki[3]))
        s10.append(float(podatki[4]))

f.close()

roll = 10


e0 = [np.average(e0[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
e1 = [np.average(e1[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
e5 = [np.average(e5[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
e10 = [np.average(e10[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]

s0 = [np.average(s0[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
s1 = [np.average(s1[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
s5 = [np.average(s5[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
s10 = [np.average(s10[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]

m0 = [np.average(m0[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
m1 = [np.average(m1[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
m5 = [np.average(m5[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
m10 = [np.average(m10[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]

c0 = [np.average(c0[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
c1 = [np.average(c1[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
c5 = [np.average(c5[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]
c10 = [np.average(c10[max(0,i-roll):min(len(s0), i+roll)]) for i in range(len(s0))]




plt.plot(t,e0,'+',label='H=0')
plt.plot(t,e1,'+',label='H=0.1')
plt.plot(t,e5,'+',label='H=0.5')
plt.plot(t,e10,'+',label='H=1')

plt.axvline(x=2.269,color='r', label = "$k_bT_c$")
plt.ylabel('energija')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('energija_tudi_zunanje.png')
plt.legend()
plt.show()


plt.plot(t,s0,'+',label='H=0')
plt.plot(t,s1,'+',label='H=0.1')
plt.plot(t,s5,'+',label='H=0.5')
plt.plot(t,s10,'+',label='H=1')


plt.axvline(x=2.269,color='r', label = "$k_bT_c$")
plt.ylabel('susceptibilnost')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('susceptibilnost_tudi_zunanje.png')
plt.legend()
plt.show()


plt.plot(t,m0,'+',label='H=0')
plt.plot(t,m1,'+',label='H=0.1')
plt.plot(t,m5,'+',label='H=0.5')
plt.plot(t,m10,'+',label='H=1')


plt.axvline(x=2.269,color='r', label = "$k_bT_c$")
plt.ylabel('magnetizacija')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('magneizacija_tudi_zunanje2.png')
plt.legend()
plt.show()


plt.plot(t,c0,'+',label='H=0')
plt.plot(t,c1,'+',label='H=0.1')
plt.plot(t,c5,'+',label='H=0.5')
plt.plot(t,c10,'+',label='H=1')

plt.axvline(x=2.269,color='r', label = "$k_bT_c$")
plt.ylabel('specifična toplota')
plt.xlabel('temperatura')
plt.grid()
plt.savefig('specificna_tudi_zunanje.png')
plt.legend()
plt.show()