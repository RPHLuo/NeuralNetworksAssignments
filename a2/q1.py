import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Sigmoid used. replace
def model(X, w_h1, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    return tf.matmul(h1, w_o)

def answer(x,y):
    return math.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y

#training: 10, testing: 9
def data(n):
    uniformRange = np.linspace(-1,1,n)
    data = []
    for i in range(0,n):
        for j in range(0,n):
            data.append([uniformRange[i],uniformRange[j]])
    return data

def runNeuralNetwork(h1Size, optimizerIndex):
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])

    size_h1 = tf.constant(h1Size, dtype=tf.int32)

    w_h1 = init_weights([2, size_h1])
    w_o = init_weights([size_h1, 1])
    py_x = model(X, w_h1, w_o)

    learningRate = 0.02
    cost = tf.sqrt(tf.losses.mean_squared_error(py_x, Y))
    gradientDescentOptimizer_train = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    momentumOptimizer_train = tf.train.MomentumOptimizer(learningRate,0.9).minimize(cost)
    RMSPropOptimizer_train = tf.train.RMSPropOptimizer(learning_rate=learningRate).minimize(cost)
    optimizers = [gradientDescentOptimizer_train, momentumOptimizer_train, RMSPropOptimizer_train]
    predict_op = py_x

    trX = data(10)
    trY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), trX))))
    trX = np.matrix(trX)
    teX = data(9)
    teY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), teX))))
    teX = np.matrix(teX)

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(10):
            sess.run(optimizers[optimizerIndex], feed_dict={X: trX[0:100], Y: trY[0:100]})
            print("training: ", i, sess.run(tf.losses.mean_squared_error(labels=trY, predictions=sess.run(predict_op, feed_dict={X: trX}))))
            print(i, sess.run(tf.losses.mean_squared_error(teY, sess.run(predict_op, feed_dict={X: teX}))))
        #done so do the contour
        #sess.run(predict_op, feed_dict={X: trX})


X,Y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
Z = np.cos(X + 6 * 0.35 * Y) + 2 * 0.35 * np.multiply(X,Y)
plt.contour(X,Y,Z)
plt.show()


sizes = [2,8,50]
for i in range(0,3):
    for j in range(0,3):
        runNeuralNetwork(sizes[i],j)
