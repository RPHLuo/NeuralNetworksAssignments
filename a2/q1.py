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

def runNeuralNetwork(h1Size, optimizerIndex, plotColour, contourSize, plot, epochs, earlyStop, record):
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

    trResults = []
    teResults = []
    vResults = []

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        vFail = 0
        for i in range(epochs):
            for start, end in zip(range(0, 90, 10), range(10, 100, 10)):
                sess.run(optimizers[optimizerIndex], feed_dict={X: trX[start:end], Y: trY[start:end]})
            if record:
                trResults.append(sess.run(tf.losses.mean_squared_error(trY, sess.run(predict_op, feed_dict={X: trX}))))
                teResults.append(sess.run(tf.losses.mean_squared_error(teY, sess.run(predict_op, feed_dict={X: teX}))))
                vResults.append(sess.run(tf.losses.mean_squared_error(vY, sess.run(predict_op, feed_dict={X: vX}))))
            if earlyStop:
                error = sess.run(tf.losses.mean_squared_error(vY, sess.run(predict_op, feed_dict={X: vX})))
                #print(error)
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
            contourInputs = np.matrix(data(contourSize))
            Xval = contourInputs[:,0]
            Yval = contourInputs[:,1]
            Z = sess.run(predict_op, feed_dict={X: contourInputs}).reshape(contourSize,contourSize)
            Xval,Yval = np.meshgrid(np.linspace(-1,1,contourSize),np.linspace(-1,1,contourSize))
            Zans = np.cos(Xval + 6 * 0.35 * Yval) + 2 * 0.35 * np.multiply(Xval,Yval)
            plt.contour(Yval, Xval,Z,colors=plotColour)
    if (record):
        return [trResults, teResults, vResults]

contourColours = ['red','blue','green']
sizes = [2,8,50]


X,Y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
Z = np.cos(X + 6 * 0.35 * Y) + 2 * 0.35 * np.multiply(X,Y)
plt.figure()
plt.contour(X,Y,Z, colors='black')

#part a
for i in range(0,3):
    runNeuralNetwork(h1Size=sizes[i], optimizerIndex=0, plotColour=contourColours[i], contourSize=100, plot=True, epochs=50000, earlyStop=False, record=False)
plt.show()

#part b
for i in range(0,3):
    for j in range(0,3):
        runNeuralNetwork(h1Size=sizes[j],optimizerIndex=i, plotColour=0, contourSize=0, plot=False, epochs=50000, earlyStop=False, record=False)
# 0 = gd, 1 = momentum, 2 = rms
CPUTimeY = []
CPUTimeX = np.arange(0,9)
for i in range(0,3):
    for j in range(0,3):
        start = time.time()
        runNeuralNetwork(h1Size=sizes[j],optimizerIndex=i, plotColour=0, contourSize=0, plot=False, epochs=100, earlyStop=False, record=False)
        end = time.time()
        elapsed = end - start
        CPUTimeY.append(elapsed)
# the bar chart shows the first 3 as GD, then Momentum then RMS
# the 1st, 4th, 7th are 2 neurons
# the 2nd, 5th, 8th are 8 neurons
# the 3rd, 6th, 9th are 50 neurons
plt.bar(CPUTimeX, CPUTimeY)
plt.show()

#part c
# 200 epochs chosen since it takes very long to put more epochs for validation
for i in range(0,3):
    runNeuralNetwork(h1Size=sizes[i], optimizerIndex=2, plotColour=0, contourSize=0, plot=False, epochs=200, earlyStop=True, record=False)
rangeSizes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 100, 1000]
for i in range(0,len(rangeSizes)):
    print(rangeSizes[i])
    runNeuralNetwork(h1Size=rangeSizes[i], optimizerIndex=2, plotColour=0, contourSize=0, plot=False, epochs=5000, earlyStop=False, record=False)

results = runNeuralNetwork(h1Size=8, optimizerIndex=2, plotColour=0, contourSize=0, plot=False, epochs=100, earlyStop=True, record=True)
plt.plot(results[0])
plt.plot(results[1])
plt.plot(results[2])
plt.show()

X,Y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
Z = np.cos(X + 6 * 0.35 * Y) + 2 * 0.35 * np.multiply(X,Y)
plt.figure()
plt.contour(X,Y,Z, colors='black')

runNeuralNetwork(h1Size=8, optimizerIndex=2, plotColour=contourColours[0],contourSize=10, plot=True, epochs=5000, earlyStop=True, record=False)
runNeuralNetwork(h1Size=8, optimizerIndex=2, plotColour=contourColours[1],contourSize=10, plot=True, epochs=5000, earlyStop=False, record=False)
plt.show()
