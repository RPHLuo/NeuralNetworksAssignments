import numpy as np
from scipy import linalg
import sympy as sp
A = np.matrix([
[3,2,-1,4],
[1,0,2,3],
[-2,-2,3,-1]
])

Anull = sp.Matrix(A).nullspace()
print("independent vectors of null space")
print(Anull[0])
print(Anull[1])

Arref = sp.Matrix(A).rref()
print("Row Echelon form of A:")
print(Arref)
# No because the reduced row echelon form indicates
# that there are only two linearly independent vectors

At = np.transpose(A)
Atrref = sp.Matrix(At).rref()
print("Row Echelon form of A transpose:")
print(Atrref)
# No because the reduced row echelon form indicates
# that there are only two linearly independent vectors
print("Null Space:")
print(linalg.null_space(A))
# No inverse, there is a pseudoinverse
invA = np.linalg.pinv(A)
print("pseudoinverse:")
print(invA)
# this kinda shows the identity matrix
print("A * A pseudoinverse:")
print(np.matmul(A,invA))
