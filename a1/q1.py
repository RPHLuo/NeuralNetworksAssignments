import numpy as np

#note u is the users and v is the movies

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
print("Alice4D: " + str(alice4D))


u, s, v = np.linalg.svd(a)
u = u[:, [0,1]]
s = np.delete(s,[2,3])
s = np.diagflat(s)
v = v[[0,1]]
alice2D = alice*u*np.linalg.inv(s)
print("Alice2D: " + str(alice2D))

prediction = np.average(alice) + (u[3] * s * v[:,0])
print("prediction: " + str(prediction[0,0]))
