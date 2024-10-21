import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import parse_zivila

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

data = parse_zivila.getData()
names = list(data.keys())
N = len(names)

#energy, fat, carb, protein, Ca, Fe, VitC, K, Na, price
#all per 100g of food

#cost function

maxMass = 200

A_lt = np.zeros((12 + len(names), N))
C = np.zeros(N)
for i in range(N):
    C[i] = data[names[i]][0] #energy

    A_lt[0,i] = -data[names[i]][1] #fat
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

    A_lt[11,i] = -data[names[i]][-1]  #B12

    #Limit single stuff
    A_lt[12+i,i] = 100 #masa

A_lt[10,:] = np.zeros(N)
A_lt[10,[6,19,43,45,46,47]] = 100



#b_lt = np.array([-70,-310,-50,-1000,-18,2000])
#b_lt = np.array([-2000,-310,-50,-1000,-18,2000, -60, -3500, -500, 2400, -70])
b_lt = [-70,-310,-50,-1000,-18,2000, -60, -3500, -500, 2400, 100, -3]
for i in range(N):
    b_lt.append(maxMass)
b_lt = np.array(b_lt)


A_eq = np.zeros((1,N))
A_eq[0,[8,9]] = 1
A_eq[0,[11, 12, 25, 38, 41, 42]] = -1


b_eq = [0]

res = linprog(C, A_lt, b_lt, A_eq, b_eq)

#DO NICE STUFF WITH SOLUTION



a,b = (3,3)
composite_list = ["Energijska\nvrednost", "Maščobe", "Ogljikovi\nhidrati", "Proteini", "Kalcij", "Železo", "Vitamin C", "Kalij", "Natrij", "Cena", "Teža", "Vitamin B12"]
units = ["kcal", "g","g", "g", "mg", "mg", "mg", "mg", "mg", "EUR", "g", "$\\mu$g"]

print(len(data[names[0]]))
print(len(composite_list))
print(len(units))

content = np.zeros((len(composite_list), N))
for i, part in enumerate(composite_list):
    if part == "Teža":
        for j in range(N):
            val = 100 * res.x[j]
            content[i,j] = val
        continue

    if part == "Vitamin B12":
        for j in range(N):
            val = data[names[j]][-1] * res.x[j]
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
    if content[-2,j] != 0:
        print(f"{names[j]}: {content[-2,j]:0.1f}")

#PLOT

for i in range(len(content[:,0])):
    if np.sum(content[i,:]) != 0:
        content[i,:] = content[i,:] / np.sum(content[i,:]) * 100

k = 0
colors = ["black", "gray", "red", "blue", "green", "purple", "orange", "lime", "pink", "goldenrod", "brown", "olive", "cyan", "gold", "skyblue", "lightseagreen", "skyblue", "skyblue", "skyblue", "skyblue", "skyblue", "skyblue"]


for j in range(N):
    ind = np.arange(len(content[:,0]))

    if np.sum(content[:,j]) == 0: continue

    if res.x[j] != 0: 
        k+=1

    plt.bar(ind, content[:,j], 0.35, bottom=np.sum(content[:,0:j], axis=1), label = names[j], color = colors[k])

plt.xticks(ind, [composite_list[i] + f"\n {np.format_float_positional(totals[i], precision=3, unique=False, fractional=False, trim='-')}{units[i]}" for i in range(len(composite_list))])
plt.ylabel("Delež[%]")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()


    
    


plt.show()













