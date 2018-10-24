import numpy as np
import tensorflow as tf
import math

#f(x,y) = cos(x + 6*0.35y) + 2*0.35xy
#x,y e [-1,1]


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    return tf.matmul(h1, w_o)

def answer(x,y):
    return math.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y

#training: 10, testing: 9
def data(n):
    uniformRange = []
    if n == 10:
        uniformRange = np.arange(-0.9,1,0.2)
    else:
        uniformRange = np.arange(-0.8,1,0.2)
    data = []
    for i in range(0,n):
        for j in range(0,n):
            data.append([uniformRange[i],uniformRange[j]])
    return data

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

size_h1 = tf.constant(8, dtype=tf.int32)

w_h1 = init_weights([2, size_h1])
w_o = init_weights([size_h1, 1])

py_x = model(X, w_h1, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

trX = data(10)
trY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), trX))))
trX = np.matrix(trX)
teX = data(9)
teY = np.transpose(np.matrix(list(map(lambda x: answer(x[0], x[1]), teX))))
teX = np.matrix(teX)

print(teY)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    print(range(0,len(trX)))
    for i in range(3):
        for start, end in zip(range(0, len(trX)), range(1, len(trX)+1)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))
