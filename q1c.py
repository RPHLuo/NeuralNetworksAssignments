import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

# Sigmoid used. replace with tanh for hyperbolic tangent
def model(X, w_h1, w_o):
    h1 = tf.nn.tanh(tf.nn.sigmoid(tf.matmul(X, w_h1)))
    return tf.matmul(h1, w_o)

def answer(x,y):
    return np.cos(x + (6 * 0.35 * y)) + (2 * 0.35 * x * y)

#training: 10, testing: 9
def data(n):
    uniformRange = np.linspace(-1,1,n)
    data = []
    for i in range(0,n):
        for j in range(0,n):
            data.append([uniformRange[i],uniformRange[j]])
    return data
def validationData(n):
    return np.random.rand(n**2,2) * 2 - 1

def runNeuralNetwork(h1Size, optimizerIndex, plotColour, plot, epochs, earlyStop):
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])

    size_h1 = tf.constant(h1Size, dtype=tf.int32)

    w_h1 = init_weights([2, size_h1])
    w_o = init_weights([size_h1, 1])
    py_x = model(X, w_h1, w_o)

    learningRate = 0.02
    tolerance = 0.02
    #cost = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(py_x, Y))))
    cost = tf.losses.mean_squared_error(py_x, Y)
    traingd = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    traingdm = tf.train.MomentumOptimizer(learningRate,0.9).minimize(cost)
    traingrms = tf.train.RMSPropOptimizer(learning_rate=learningRate).minimize(cost)
    optimizers = [traingd, traingdm, traingrms]
    predict_op = py_x

    trX = data(10)
    trY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), trX))))
    trX = np.matrix(trX)
    teX = data(9)
    teY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), teX))))
    teX = np.matrix(teX)

    vX = np.matrix([])
    vY = np.matrix([])
    vError = 0
    #early stopping
    if (earlyStop):
        vX = validationData(10)
        vY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), vX))))
        vX = np.matrix(vX)
        vError = np.inf

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        vFail = 0
        for i in range(epochs):
            for start, end in zip(range(0, 90, 10), range(10, 100, 10)):
                sess.run(optimizers[optimizerIndex], feed_dict={X: trX[start:end], Y: trY[start:end]})
            if earlyStop:
                error = sess.run(tf.losses.mean_squared_error(vY, sess.run(predict_op, feed_dict={X: vX})))
                print(error)
                if (error >= vError):
                    vFail += 1
                else:
                    vError = error
                if (vFail >= 10):
                    print("Validation early stopping")
                    break
            if i % (epochs/25) == 0:
                error = sess.run(tf.losses.mean_squared_error(trY, sess.run(predict_op, feed_dict={X: trX})))
                #print(error
                if (error < tolerance):
                    print("converged below tolerance")
                    break
        print(i, sess.run(tf.losses.mean_squared_error(teY, sess.run(predict_op, feed_dict={X: teX}))))
        #done so do the contour
        if plot:
            contourInputs = np.matrix(data(10))
            Xval = contourInputs[:,0]
            Yval = contourInputs[:,1]
            Z = sess.run(predict_op, feed_dict={X: contourInputs}).reshape(10,10)
            Xval,Yval = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
            Zans = np.cos(Xval + 6 * 0.35 * Yval) + 2 * 0.35 * np.multiply(Xval,Yval)
            plt.contour(Yval, Xval,Z,colors=plotColour)


#X,Y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
#Z = np.cos(X + 6 * 0.35 * Y) + 2 * 0.35 * np.multiply(X,Y)
#plt.figure()
#plt.contour(X,Y,Z, colors='black')

#sizes = [2,8,50]
contourColours = ['red','blue','green']

#part a
#for i in range(0,3):
#    runNeuralNetwork(sizes[i], 0, contourColours[i], True, 50000, False)
#plt.show()

#part b
#for i in range(0,3):
#    for j in range(0,3):
#        runNeuralNetwork(sizes[j],i, contourColours[j], False, 50000, False)
# 0 = gd, 1 = momentum, 2 = rms
#for i in range(0,3):
#    for j in range(0,3):
#        start = time.time()
#        runNeuralNetwork(sizes[j],i, contourColours[j], False, 100, False)
#        end = time.time()
#        elapsed = end - start
#        print("optimizer: ", i,"\nsize: ",sizes[j], "\ntime: " ,elapsed)

#part c
# 200 epochs chosen since it takes very long to put more epochs for validation
#for i in range(0,3):
#    runNeuralNetwork(sizes[i], 2, contourColours[i], False, 200, True)
#rangeSizes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 100, 1000]
#for i in range(0,len(rangeSizes)):
#    print(rangeSizes[i])
#    runNeuralNetwork(rangeSizes[i], 2, contourColours[0], False, 5000, False)


X,Y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
Z = np.cos(X + 6 * 0.35 * Y) + 2 * 0.35 * np.multiply(X,Y)
plt.figure()
plt.contour(X,Y,Z, colors='black')

runNeuralNetwork(8, 2, contourColours[0], True, 5000, True)
runNeuralNetwork(8, 2, contourColours[1], True, 5000, False)
plt.show()
