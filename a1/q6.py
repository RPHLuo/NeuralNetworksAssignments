import numpy as np
from mnist import MNIST

A = [[1,2,3],
     [4,5,6],
     [7,8,9]]

B = [[1,1,1],
     [2,2,2],
     [3,3,3]]

i = [[3,6,2],
     [9,8,2],
     [5,2,1]]

def digitCalc (unknown, digitData):
    concat = digitData
    z = np.matrix(unknown).flatten()
    bases = 0    

    for digit in digitData:   
        digit = np.matrix(digit).flatten().T

        if bases == 0 :
            concat = digit
            bases += 1
            continue
        
        concat = np.hstack((concat, digit))
        bases += 1

    u,s,vT = np.linalg.svd(concat)

    rank = np.linalg.matrix_rank(u)
    
    I = np.identity(rank)

    X = np.subtract(I, np.matmul(u,u.T))

    return np.linalg.norm(np.matmul(X,z.T))

database = list()
database.append([A,B,A,A,B])
database.append([B,B,A,B])

values = list()

for digitData in database:
    values.append(digitCalc(i, digitData))
    print(digitCalc(i, digitData))

print(min(values))
#print(digitCalc(i,(A,B,A,A,B,B,A)))
