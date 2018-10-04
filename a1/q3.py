import numpy as np

#create A using the function provided
A = np.fromfunction(lambda i, j: (1 - ((-0.7 + (0.001 * i)) ** 2) - ((-0.7 + (0.001 * j)) ** 2)), (1400, 1400))

#take the SVD of A
u, s, vt = np.linalg.svd(A)

vtk = vt[[0,1]] #Take the first 2 rows of vT
uk = u[:, [0,1]] #take the first 2 columns of u
sk = np.diagflat(s[0:2]) #take the first 2 significant values and convert them to a 2x2 matrix

#A2 is the sum of the 3 components above
A2 = np.matmul((np.matmul(uk, sk)), vtk)

#Prints the difference between our original matrix and its rank-2 approximation (A2)
print(np.linalg.norm(np.subtract(A, A2)))
