from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

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

def runNeuralNetwork(data, size):
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])

    size_h1 = tf.constant(size, dtype=tf.int32)

    w_h1 = init_weights([2, size_h1])
    w_o = init_weights([size_h1, 1])
    py_x = model(X, w_h1, w_o)

    learningRate = 0.02
    tolerance = 0.02
    cost = tf.losses.mean_squared_error(py_x, Y)
    traingdm = tf.train.MomentumOptimizer(learningRate,0.9).minimize(cost)
    predict_op = py_x

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(epochs):
            for start, end in zip(range(0, 90, 10), range(10, 100, 10)):
                sess.run(optimizers[optimizerIndex], feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, sess.run(tf.losses.mean_squared_error(teY, sess.run(predict_op, feed_dict={X: teX}))))

runFeedforeward()
