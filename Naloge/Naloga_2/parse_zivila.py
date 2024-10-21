import numpy as np

def getData():
    with open("Naloga_2\\zivila.txt", mode = "r") as f:
        lines = f.readlines()[1:]

    data = dict()

    for i, line in enumerate(lines):
        name, energy, fat, carb, protein, Ca, Fe, VitC, K, Na, price = line.strip().replace("\t\t", "\t").strip().split("\t")

        price, B12 = price.split("   ")

        
        data[name] = np.array([float(energy), float(fat), float(carb), float(protein), float(Ca), float(Fe), float(VitC), float(K), float(Na), float(price), float(B12)])
    
    return data

