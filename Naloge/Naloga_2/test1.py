import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import parse_zivila

C = [-10,-2]

A = np.zeros((2,2))
A[0,:] = [1,1]
A[1,:] = [10,3]

b = [1,6]

res = linprog(C, A, b)

print(C)
print(A)
print(b)
print(res.x)