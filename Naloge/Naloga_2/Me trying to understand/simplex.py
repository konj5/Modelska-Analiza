import numpy as np

def simplex(table):
    def bottom_is_negative(table):
        for x in table[-1,:]:
            if x < 0:
                return True
        return False

    while bottom_is_negative(table):
        pivot_collumn = np.argmin(table[-1,:])
        pivot_line = np.argmin(table[:-1,-1]/table[:-1,pivot_collumn])

        table[pivot_line,:] = table[pivot_line,:] / table[pivot_line,pivot_collumn]


        for i in range(len(table[:,pivot_collumn])):
            if i == pivot_line:
                continue
            
            table[i,:] = table[i,:] - table[pivot_line,:] * table[i,pivot_collumn]

        

    ###Get results

    def is_basic(array):
        had_1 = False
        for x in array:
            if x == 0:
                continue
            if x == 1 and had_1 == False:
                had_1 = True
                continue
            else:
                return False
            
        return True

    def find_first_1(array):
        for i, x in enumerate(array):
            if x == 1:
                return i

    solutions = []
    for i in range(len(table[0,:])-2):
        if is_basic(table[:,i]):
            solutions.append(table[find_first_1(table[:,i]), -1])
        else:
            solutions.append(0)

    return solutions, table
