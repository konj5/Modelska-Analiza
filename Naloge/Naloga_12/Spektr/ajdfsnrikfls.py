import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from nitime import algorithms as alg

def aproksimacija(podatki,seznam_koeficientov,iteracij):
    for i in range(iteracij):
        vsota=0
        for j in range(len(seznam_koeficientov)):
            vsota+=podatki[-j-1]*seznam_koeficientov[j]
        podatki.append(vsota)
    return podatki



x=[]

f = open('Naloge\\Naloga_12\\val2.dat', 'r',encoding='utf-8')
data_file=f.readlines()

for vrsta in data_file:
    x.append(float(vrsta))



#odvisnost od stevila polov
#
dolzina=10
a,b=alg.AR_est_YW(np.array(x),dolzina)
f10=alg.autoregressive.AR_psd(a,b)
#
dolzina=20
a,b=alg.AR_est_YW(np.array(x),dolzina)
f20=alg.autoregressive.AR_psd(a,b)
#
dolzina=50
a,b=alg.AR_est_YW(np.array(x),dolzina)
f50=alg.autoregressive.AR_psd(a,b)
#
dolzina=100
a,b=alg.AR_est_YW(np.array(x),dolzina)
f100=alg.autoregressive.AR_psd(a,b)
#
dolzina=200
a,b=alg.AR_est_YW(np.array(x),dolzina)
f200=alg.autoregressive.AR_psd(a,b)
#
#fft:
fft_koncentracija=abs(np.fft.fft(x))**2
fft_koncentracija=fft_koncentracija/sum(fft_koncentracija)
f_cas=np.arange(len(fft_koncentracija))/len(fft_koncentracija)
#
plt.plot(f_cas,fft_koncentracija,label='FFT')
plt.plot(f10[0]/2/np.pi,f10[1]/sum(f10[1]),label='10')
plt.plot(f20[0]/2/np.pi,f20[1]/sum(f20[1]),label='20')
plt.plot(f50[0]/2/np.pi,f50[1]/sum(f50[1]),label='50')
plt.plot(f100[0]/2/np.pi,f100[1]/sum(f100[1]),label='100')
plt.plot(f200[0]/2/np.pi,f200[1]/sum(f200[1]),label='200')
plt.legend(title='koeficientov')
plt.yscale('log')
plt.xlabel('frekvenca')
plt.ylabel('gostota moči')
plt.xlim(xmax=1/2)
plt.grid()
plt.savefig('val2_primerjava_fft.png')
plt.show()
#
#
plt.plot(f_cas,fft_koncentracija,label='FFT')
plt.plot(f10[0]/2/np.pi,f10[1]/sum(f10[1]),label='10')
plt.plot(f20[0]/2/np.pi,f20[1]/sum(f20[1]),label='20')
plt.plot(f50[0]/2/np.pi,f50[1]/sum(f50[1]),label='50')
plt.plot(f100[0]/2/np.pi,f100[1]/sum(f100[1]),label='100')
plt.plot(f200[0]/2/np.pi,f200[1]/sum(f200[1]),label='200')
plt.legend(title='koeficientov')
plt.yscale('log')
plt.xlabel('frekvenca')
plt.ylabel('gostota moči')
plt.xlim(xmax=1/2)
plt.grid()
plt.savefig('val2_primerjava_koef.png')
plt.show()




polovica=int(len(x)/2)

podatki=x[:polovica]
st_koef=20


a,b=alg.AR_est_YW(np.array(podatki),st_koef)


# y2=aproksimacija(x,a,ii)
y3=aproksimacija(podatki,a,polovica)


plt.plot(x,label='meritev')
plt.plot(y3,label='napoved N='+str(st_koef))
plt.legend(title='val2.dat')
plt.grid()
plt.axvline(x=polovica,color='r')
plt.xlim(xmax=len(y3))
plt.savefig('napoved_val2.png')
plt.show()