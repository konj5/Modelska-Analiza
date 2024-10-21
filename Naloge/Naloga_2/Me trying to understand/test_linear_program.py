import numpy as np
import simplex

np.set_printoptions(precision=2)

# P = 10 x1 + 2 x2, x + y >= 1 in 10x + 3y >= 6


vals = 2

cost = [10,2]

rels_eq = []
rels_le = []
rels_ge = []

rels_ge.append([1,1,1])
rels_le.append([10,3,6])


if len(rels_ge)  == 0:
    ####CREATE TABLE
    Ny = len(rels_eq) + len(rels_le) + len(rels_ge) + 1
    Nx = vals + len(rels_le) + len(rels_ge) + 2

    table = np.zeros((Ny, Nx), dtype=float)

    i = 0
    for rel in rels_eq:
        table[i, 0:vals] = rel[:-1]
        table[i, -1] = rel[-1]
        i+=1

    si = 0
    for rel in rels_le:
        table[i, 0:vals] = rel[:-1]
        table[i, vals+si] = 1
        table[i, -1] = rel[-1]
        i+=1
        si+=1

    table[i, 0:vals] = -np.array(cost)
    table[i, -2] = 1

    print(table)

    solutions, end_table = simplex.simplex(table)
    print(end_table)
    print(solutions)

else:
     ####CREATE TABLE
    Ny = len(rels_eq) + len(rels_le) + len(rels_ge) + 1
    Nx = vals + len(rels_le) + 2*len(rels_ge) + 2

    table = np.zeros((Ny, Nx), dtype=float)

    i = 0
    for rel in rels_eq:
        table[i, 0:vals] = rel[:-1]
        table[i, -1] = rel[-1]
        i+=1

    si = 0
    for rel in rels_le:
        table[i, 0:vals] = rel[:-1]
        table[i, vals+si] = 1
        table[i, -1] = rel[-1]
        i+=1
        si+=1

    for rel in rels_ge:
        table[i, 0:vals] = rel[:-1]
        table[i, vals+si] = -1
        table[i, vals+si+len(rels_ge)] = 1
        table[i, -1] = rel[-1]
        i+=1
        si+=1

    table[i, 0:vals] = -np.array(cost)
    table[i, -2] = 1


    ### SOLVE THE artifician variable problem

    table2 = np.copy(table)
    table2[-1,:] = np.zeros(len(table2[-1,:]))
    table2[-1, -2-len(rels_ge):-1] = np.ones(len(rels_ge)+1)


    def find_first_1(array):
        for i, x in enumerate(array):
            if x == 1:
                return i
    
    #fix to be solvable
    for i in range(-2-len(rels_ge), -2):
        table2[-1,:] = table2[-1,:] - table2[find_first_1(table2[:,i]),:]

    solutions, end_table2 = simplex.simplex(table2)


    ### Use this on original problem

    mask = [True for i in range(len(end_table2[0,:]))]
    for i in range(-2-len(rels_ge), -2):
        mask[i] = False

    table = end_table2[:, mask]

    #add old cost function
    table[-1, 0:vals] = -np.array(cost)
    table[-1, -2] = 1
    

    #fix unit vectors broken by this
    for i, sol in enumerate(solutions):
        if sol == 0: continue

        row_i = find_first_1(table[:,i])

        table[-1, :] = table[-1, :] - table[row_i,:] * table[-1, i]

    solutions, end_table = simplex.simplex(table)

    print(solutions)
    print(end_table)


