import numpy as np

a = np.matrix([
[3,1,2,3],
[4,3,4,3],
[3,2,1,5],
[1,6,5,2]
])
alice = np.array([5,3,4,4])

u, s, v = np.linalg.svd(a)
v = np.transpose(v)
s = np.diagflat(s)
alice4D = alice*u*s
print(alice4D)

u, s, v = np.linalg.svd(a)
u = np.delete(u,[2,3],1)
s = np.delete(s,[2,3])
s = np.diagflat(s)
v = np.transpose(v)
v = np.delete(v,[2,3],1)
#print(u)
#print(s)
#print(v)
alice2D = alice*u*np.linalg.inv(s)
print(alice2D)
