import numpy as np
import random
from mnist import MNIST

mndata = MNIST('C:\Users\colin.daniel\Documents\School\FALL 2018\COMP 4107') # Change this to the directory where you're storing the test and training data

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
def guess(testIndex):
    values = list()
    for img in sigImgs:
        values.append(calculateResidual(testImgs[testIndex], img))

    #print("Image was: " + str(testLbls[testIndex]))
    #print("We guessed: " + str(values.index(min(values))))

    if testLbls[testIndex] == values.index(min(values)) :
        return 1
    else :
        return 0

# Calculates the residual between the given image data of a digit and our unknown digit data
def calculateResidual (unknownDigit, digitData):
    u,s,vT = np.linalg.svd(digitData)

    rank = np.linalg.matrix_rank(u)
    
    I = np.identity(rank)

    X = np.subtract(I, np.matmul(u,u.T))

    return np.linalg.norm(np.matmul(X, np.transpose(unknownDigit)), 2)


print("Program Start")
sigImgs = generateImages(10)
correct = 0

for i in range(10):
    index = random.randrange(0, len(testImgs))
    correct += guess(index)

print("Number Correct: " + str(correct))
print("Program End")
