import numpy as np

a = [
[1,2,3],
[2,3,4],
[4,5,6],
[1,1,1]
]

u,d,v = np.linalg.svd(a)

print(u)
print(d)
print(v)
