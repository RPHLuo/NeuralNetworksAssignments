import numpy as np

a = [
[1,2,3],
[2,3,4],
[4,5,6],
[1,1,1]
]

u,s,v = np.linalg.svd(a)

print("U:\n" + str(u))
print("S:\n" + str(s))
print("V:\n" + str(v))
