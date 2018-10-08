import numpy as np
import random
import matplotlib.pyplot as plt
from mnist import MNIST

mndata = MNIST('$$$') # Set this to the directory path where you're storing the test and training data

images, labels = mndata.load_training()
testImgs, testLbls = mndata.load_testing()

# Generates the image data for all digits (0 - 9)
def generateImages(rank):
    imageList = list()
    for i in range(10):
        imageList.append(getImageData(i, rank))

    return imageList


# Given a digit and a number of training images to sample from (basis)
# This function builds a m x n matrix - where m is the rank of the image vector
# and n is the number of images used to form the basis
def getImageData(digit, basis):
    index = 0
    nSet = []
    count = 0 # Used to track how many images we've added

    while count < basis:
        # Gets random samples from the image dataset
        index = random.randrange(0, len(testImgs))
        image = images[index]
        
        if labels[index] == digit:
            if count == 0:
                nSet = np.transpose(image)
            else:
                # Concatenates any subsequent image column vectors onto the matrix
                nSet = np.column_stack((nSet, np.transpose(image)))

            count += 1

        index += 1
    return nSet


# Compares the residuals returned over all 10 digits and chooses the lowest one
# Returns 1 if correct and 0 if incorrect
def guess(testIndex, sigImgs):
    values = list()
    for img in sigImgs:
        values.append(calculateResidual(testImgs[testIndex], img))

    if testLbls[testIndex] == values.index(min(values)) :
        return 1
    else :
        return 0

# Calculates the residual between the given image data of a digit and our unknown digit data
def calculateResidual (unknownDigit, digitData):    
    u,s,vT = np.linalg.svd(digitData)
    
    rank = np.linalg.matrix_rank(u)
    A = np.subtract(np.identity(rank), np.matmul(u, u.T))
    AT = np.transpose(A)
    z = unknownDigit

    change = np.matmul(AT, np.matmul(A, z))
    return np.linalg.norm(change, 2)

# Runs tests for a specified number of basis images
def runTest(basisNum):
    sigImgs = generateImages(basisNum)
    correct = 0

    for i in range(10):
        index = random.randrange(0, len(testImgs))
        correct += guess(index, sigImgs)

    basisNumbers.append(basisNum)
    percentCorrect.append(correct * 10)





basisNumbers = list()
percentCorrect = list()

print("MAKE SURE YOU DOWNLOAD THE MNIST DATA AND POINT TO ITS LOCATION (SEE LINE 6)")
runTest(100)
print("Test 1 Complete")
runTest(250)
print("Test 2 Complete")
runTest(500)
print("Test 3 Complete")
runTest(750)
print("Test 4 Complete")
runTest(1000)
print("Test 5 Complete")
print("All Tests Complete")

plt.plot(basisNumbers, percentCorrect)
plt.xlabel('Number of Basis Images Used')
plt.ylabel('Classification Percentage')
plt.show()
