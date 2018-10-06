import numpy as np

x  = [0.03, 0.02, 0.01]

A = [[1, 2, 3],
     [2, 3, 4],
     [4, 5, 6],
     [1, 1, 1]]

AT = np.transpose(A)

b = [1, 1, 1, 1]

def gradientDescent (x, stepSize, tolerance):
    while (1):
        change = np.matmul(AT, np.matmul(A, x)) - np.matmul(AT, b)
        if np.linalg.norm(change) <= tolerance: break
        
        x = np.subtract(x, stepSize * change)

    print(np.dot(x,A[0]))
    print(np.dot(x,A[1]))
    print(np.dot(x,A[2]))
    print(np.dot(x,A[3]))
    
gradientDescent(x, 0.01, 0.01)
        
