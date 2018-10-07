import numpy as np

w  = [0.03, 0.02, 0.01] # The weight vector 
oldW = list(w) # This will be compared to the new weight vector every iteration

A = [[1, 2, 3],
     [2, 3, 4],
     [4, 5, 6],
     [1, 1, 1]]

target = 1 # The target vector (Represented as a value since every vector component is the same)
rate = 0.01 # The learning rate

count = 1 # Tracks the number of iterations taken to solve

while (1) :
    # Loops through each component of the weight vector to correct each individually
    for i in range(3):
        deltaW = 0
        # Loops through each "trial" in our training set 
        for j in range(4):
            deltaW += A[j][i] * (target - (np.dot(w, A[j])))
            j += 1
        w[i] += rate * deltaW
        i += 1

    # If the weights do not change, we've reached the minimum and can exit the loop
    if np.allclose(oldW, w) : break

    oldW = list(w)
    count += 1

print("Number of Iterations: " + str(count))
print("Final Weights:")
print(w)

print("Final Outputs:")
print(np.dot(w, A[0]))
print(np.dot(w, A[1]))
print(np.dot(w, A[2]))
print(np.dot(w, A[3]))

