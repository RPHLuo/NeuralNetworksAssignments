import numpy as np
import math
import random
from astropy.table import Table, Column

A = [[1, 2, 3],
     [2, 3, 4],
     [4, 5, 6],
     [1, 1, 1]]

AT = np.transpose(A)

b = [1, 1, 1, 1]

# These store the values of x and number of iterations over each trial (for the table)
xList = list()
countList = list()

# Generates a starting x vector with random numbers normally distributed around the origin
def generateRandomX():
    x = list()
    for i in range (3):
        x.append(random.normalvariate(0, 2))
    return x

def gradientDescent (stepSize, tolerance):
    x = generateRandomX()
    
    count = 0
    while (1):
        change = np.matmul(AT, np.matmul(A, x)) - np.matmul(AT, b)
        if math.isinf(np.linalg.norm(change, 2)):
            x = "Could Not Converge (step size too large)"
            break
        if np.linalg.norm(change, 2) <= tolerance:
            break

        x = np.subtract(x, stepSize * change)
        count += 1
        
    xList.append(str(x))
    countList.append(str(count))


gradientDescent(0.01, 0.01)
gradientDescent(0.05, 0.01)
gradientDescent(0.1, 0.01)
gradientDescent(0.15, 0.01)
gradientDescent(0.2, 0.01)
gradientDescent(0.25, 0.01)
gradientDescent(0.5, 0.01)



t = Table()
t['Values of x'] = xList
t['Number of Iterations'] = countList
print(t)
