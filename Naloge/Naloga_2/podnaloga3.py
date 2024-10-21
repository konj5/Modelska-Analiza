import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import parse_zivila

data = parse_zivila.getData()
names = list(data.keys())
N = len(names)

#energy, fat, carb, protein, Ca, Fe, VitC, K, Na, price
#all per 100g of food

#cost function

A_lt = np.zeros((11, N))
C = np.zeros(N)
for i in range(N):
    C[i] = data[names[i]][-1] #cena

    A_lt[0,i] = -data[names[i]][0] #energy
    A_lt[10,i] = -data[names[i]][1] #fat
    A_lt[1,i] = -data[names[i]][2] #carb
    A_lt[2,i] = -data[names[i]][3] #prot
    A_lt[3,i] = -data[names[i]][4] #ca
    A_lt[4,i] = -data[names[i]][5] #Fe

    A_lt[5,i] = 100 #masa

    #Optional
    A_lt[6,i] = -data[names[i]][6]  #vit C
    A_lt[7,i] = -data[names[i]][7]  #K
    A_lt[8,i] = -data[names[i]][8]  #Na min
    A_lt[9,i] = data[names[i]][8]  #Na max


#b_lt = np.array([-70,-310,-50,-1000,-18,2000])
b_lt = np.array([-2000,-310,-50,-1000,-18,2000, -60, -3500, -500, 2400, -70])

res = linprog(C, A_lt, b_lt)

#DO NICE STUFF WITH SOLUTION

a,b = (3,3)
composite_list = ["Energijska\nvrednost", "Maščobe", "Ogljikovi\nhidrati", "Proteini", "Kalcij", "Železo", "Vitamin C", "Kalij", "Natrij", "Cena", "Teža"]
units = ["kcal", "g","g", "g", "mg", "mg", "mg", "mg", "mg", "EUR", "g"]

content = np.zeros((len(composite_list), N))
for i, part in enumerate(composite_list):
    if part == "Teža":
        for j in range(N):
            val = 100 * res.x[j]
            content[i,j] = val
        continue

    for j in range(N):
        val = data[names[j]][i] * res.x[j]
        content[i,j] = val

totals = []
print("TOTALS:\n")
for i, part in enumerate(composite_list):
    totals.append(np.sum(content[i,:]))
    print(f"{part}: {np.sum(content[i,:])}")

print("\n\nCONTENTS:\n")
for j in range(N):
    if content[-1,j] != 0:
        print(f"{names[j]}: {content[-1,j]:0.1f}")

#PLOT

for i in range(len(content[:,0])):
    if np.sum(content[i,:]) != 0:
        content[i,:] = content[i,:] / np.sum(content[i,:]) * 100

for j in range(N):
    ind = np.arange(len(content[:,0]))

    if np.sum(content[:,j]) == 0: continue

    plt.bar(ind, content[:,j], 0.35, bottom=np.sum(content[:,0:j], axis=1), label = names[j])

plt.xticks(ind, [composite_list[i] + f"\n {np.format_float_positional(totals[i], precision=3, unique=False, fractional=False, trim='-')}{units[i]}" for i in range(len(composite_list))])
plt.ylabel("Delež[%]")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()


    
    


plt.show()













