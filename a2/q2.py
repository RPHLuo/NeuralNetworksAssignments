import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_o):
    h1 = tf.log_sigmoid(tf.matmul(X, w_h1))
    return tf.log_sigmoid(tf.matmul(h1, w_o))

def shuffle(data, order):
    newData = []
    for i in order:
        newData.append(data[i])
    return newData

def percentCorrect(guess, answer, size):
    correct = 0.0
    for index in range(0,size):
        if guess[index] == answer[index]:
            correct += 1
    return correct/size

def corrupt(data, corruption):
    for letter in data:
        randomBits = np.random.randint(0,len(data),corruption)
        for bit in randomBits:
            if (bit == 0):
                bit = 1
            else:
                bit = 0
    return data

#A-Z + jinyu
data = [
"00100010100101010001111111000110001",
"11110100011000111110100011000111110",
"01110100011000010000100001000101110",
"11110100011000110001100011000111110",
"11111100001000011110100001000011111",
"11111100001000011110100001000010000",
"01110100011000010000100111000101110",
"10001100011000111111100011000110001",
"01110001000010000100001000010001110",
"11111001000010000100001001010001000",
"10001100101010011000101001001010001",
"10000100001000010000100001000011111",
"10001110111010110001100011000110001",
"10001110011100110101100111001110001",
"01110100011000110001100011000101110",
"11110100011000111110100001000010000",
"01110100011000110001101011001001101",
"11110100011000111110101001001010001",
"01110100010100000100000101000101110",
"11111001000010000100001000010000100",
"10001100011000110001100011000101110",
"10001100011000110001100010101000100",
"10001100011000110001101011101110001",
"10001100010101000100010101000110001",
"10001100010101000100001000010000100",
"11111000010001000100010001000011111",
"00010000000001000010010100101000100",
"00100000000010000100001000010000100",
"01110100011000110001100011000110001",
"10001100011000101010001000100010000",
"10010100101001010010100101001001101"]

answerArray = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',
'Z','j','i','n','y','u']

answer = []
for i in range(0,31):
    answer.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    answer[i][i] = 1

refinedData = []

for i in range(0,len(data)):
    refinedData.append([])
    for bit in data[i]:
        refinedData[i].append(float(bit))
def TrainNetwork(epochs, corruption, testingCorruption, neurons, XData, YData, testSize, Qone, Qtwo):
    corruptedData = corrupt(XData,corruption)

    teX = XData
    teY = YData
    randomL = np.random.randint(31, size=testSize)
    for i in range(0,testSize):
        teX.append(XData[randomL[i]])
        teY.append(YData[randomL[i]])
    teX = corrupt(teX, testingCorruption)
    order = np.arange(testSize)
    np.random.shuffle(order)
    teX = np.matrix(shuffle(teX, order))
    teY = np.matrix(shuffle(teY, order))

    trX = XData
    trY = YData

    order = np.arange(31)
    np.random.shuffle(order)
    trX = np.matrix(shuffle(trX, order))
    trY = np.matrix(shuffle(trY, order))

    coX = corruptedData
    coY = YData

    order = np.arange(31)
    np.random.shuffle(order)
    coX = np.matrix(shuffle(coX, order))
    coY = np.matrix(shuffle(coY, order))

    size_h1 = tf.constant(neurons, dtype=tf.int32)

    X = tf.placeholder("float", [None, 35])
    Y = tf.placeholder("float", [None, 31])

    w_h1 = init_weights([35, size_h1]) # create symbolic variables
    w_o = init_weights([size_h1, 31])

    py_x = model(X, w_h1, w_o)

    #cost = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(py_x, Y))))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.AdamOptimizer(0.01).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(epochs):
            for start, end in zip(range(0, 30), range(1, 31)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if Qone:
            guess = sess.run(predict_op, feed_dict={X: teX})
            answer = np.argmax(teY, axis=1)
            correct = percentCorrect(guess,answer,31)
            print("neurons: ", neurons, " accuracy: ",correct)
            return correct
        elif Qtwo:
            for i in range(epochs):
                for start, end in zip(range(0, 30), range(1, 31)):
                    sess.run(train_op, feed_dict={X: coX[start:end], Y: coY[start:end]})
            for i in range(epochs):
                for start, end in zip(range(0, 30), range(1, 31)):
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            guess = sess.run(predict_op, feed_dict={X: teX})
            answer = np.argmax(teY, axis=1)
            correct = percentCorrect(guess,answer,31)
            #return error instead of accuracy
            return 1 - correct
#part a
result = []
for i in range(5,26):
    result.append(TrainNetwork(epochs=200, corruption=3, testingCorruption=3, neurons=i, XData=refinedData, YData=answer, testSize=1000, Qone=True, Qtwo=False))

plt.plot(np.arange(5,26),result)
plt.show()

#part b
errors = []
for i in range(1,300):
    print(i)
    errors.append(TrainNetwork(epochs=i, corruption=3, testingCorruption=3, neurons=15, XData=refinedData, YData=answer, testSize=1000, Qone=False, Qtwo=True))

plt.plot(np.arange(1,300),errors)
plt.show()

#part c
noiseFree = []
noiseTrained = []
for i in range(0,3):
    noiseFree.append(TrainNetwork(epochs=200, corruption=0, testingCorruption=i, neurons=15, XData=refinedData, YData=answer, testSize=1000, Qone=False, Qtwo=True))
    noiseTrained.append(TrainNetwork(epochs=200, corruption=3, testingCorruption=i, neurons=15, XData=refinedData, YData=answer, testSize=1000, Qone=False, Qtwo=True))

plt.plot(np.arange(0,3),noiseFree)
plt.plot(np.arange(0,3),noiseTrained)
plt.show()
