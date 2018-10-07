import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
import time

#Values
datasetSize = 100000
x = [0.8,0.5,0.2]
k = 14
basis = 600
basisSpeed = [600,650,700,750,800,850,900,943]
addRatings = 50

def MAE(AdfTest, U, S, VS):
    ATest = AdfTest.values
    US = np.matmul(U, S)
    count = 0
    error = 0.0
    # a is the utility matrix with the correct original values for rating a = [userId,movieId,rating]
    for a in ATest:
        if a[0] < len(US) and a[1] < len(VS):
            error += abs(a[2] - US[int(a[0]-1),:].dot(VS[:,int(a[1]-1)]))
            count += 1
    print(str(len(US)) + ": " + str(error/count))

def Speedtest(AdfTest, U, S, VS):
    ATest = AdfTest.values
    US = np.matmul(U, S)
    # a is the utility matrix with the correct original values for rating a = [userId,movieId,rating]
    for a in ATest:
        if a[0] < len(US) and a[1] < len(VS):
            #Just set the result. no need to print it
            res = abs(a[2] - US[int(a[0]-1),:].dot(VS[:,int(a[1]-1)]))

def trainSpeed(Adf,x,b):
    startTime = time.time()
    #Arbitrary random seed 3
    AdfTrain, AdfTest = train_test_split(Adf,train_size=x, test_size=1-x, random_state=3)

    AdfTrain = AdfTrain.pivot(index=0,columns=1)
    AdfTrain = AdfTrain.fillna(AdfTrain.mean())
    A = AdfTrain.values

    ABasis = np.delete(A, range(b, len(A)), 0)
    AIncrements = np.delete(A, range(0, b), 0)

    Ub,Sb,Vb = np.linalg.svd(ABasis)

    Uk = Ub[:, range(0,k)]
    Sk = np.diagflat(Sb[0:k])
    Vk = Vb[range(0,k)]

    Ssqrt = sp.linalg.sqrtm(Sk)
    VS = np.matmul(Ssqrt,Vk)

    count = 0

    for Ai in AIncrements:
        UAdd = Vk.dot(Ai).dot(np.linalg.inv(Sk))
        Uk = np.vstack((Uk , UAdd))
        count+=1
    Speedtest(AdfTest, Uk, Ssqrt, VS)
    endTime = time.time()
    duration = endTime-startTime
    totalPredictions = datasetSize * (1-x)
    print("speed when x: " + str(x) + " and basis: " + str(b) + " prediction/s: " + str(totalPredictions/duration))


def train(Adf, x):
    print("results when x is " + str(x))
    #Arbitrary random seed 3
    AdfTrain, AdfTest = train_test_split(Adf,train_size=x, test_size=1-x, random_state=3)

    AdfTrain = AdfTrain.pivot(index=0,columns=1)
    AdfTrain = AdfTrain.fillna(AdfTrain.mean())
    A = AdfTrain.values

    ABasis = np.delete(A, range(basis, len(A)), 0)
    AIncrements = np.delete(A, range(0, basis), 0)

    Ub,Sb,Vb = np.linalg.svd(ABasis)

    Uk = Ub[:, range(0,k)]
    Sk = np.diagflat(Sb[0:k])
    Vk = Vb[range(0,k)]

    Ssqrt = sp.linalg.sqrtm(Sk)
    VS = np.matmul(Ssqrt,Vk)

    count = 0
    MAE(AdfTest, Uk, Ssqrt, VS)
    for Ai in AIncrements:
        UAdd = Vk.dot(Ai).dot(np.linalg.inv(Sk))
        Uk = np.vstack((Uk , UAdd))
        count+=1
        if (count == addRatings):
            MAE(AdfTest, Uk, Ssqrt, VS)
            count = 0
    MAE(AdfTest, Uk, Ssqrt, VS)

a = np.loadtxt('./ml-100k/u.data', usecols=[0,1,2]).tolist()
Adf = pd.DataFrame(a)
for xi in x:
    train(Adf, xi)

#does not print. Only gets the value
for xi in x:
    for b in basisSpeed:
        trainSpeed(Adf,xi,b)
