import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

total_batches = 5
batch_size = 128
test_size = 256

def init_weights(shape, num):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, name='random'), name='weight'+num)

def getData(batch_number=1):
    file = './cifar-10-batches-py/data_batch_{0}'.format(batch_number)
    dict = unpickle(file)
    data = dict['data']
    labels = dict['labels']

    reshaped_x = data.reshape(len(data),3,32,32).transpose(0, 2, 3, 1)
    reshaped_y = np.zeros((len(labels),10))
    for i in range(0,len(labels)):
        reshaped_y[i][labels[i]] = 1.
    return reshaped_x, reshaped_y

def getTestData():
    file = './cifar-10-batches-py/test_batch'
    dict = unpickle(file)
    data = dict['data']
    labels = dict['labels']

    reshaped_x = data.reshape(len(data),3,32,32).transpose(0, 2, 3, 1)
    reshaped_y = np.zeros((len(labels),10))
    for i in range(0,len(labels)):
        reshaped_y[i][labels[i]] = 1.
    return reshaped_x, reshaped_y

def model(X, w, w2, w_fc, w_o, p_keep_conv, p_keep_hidden):
    with tf.variable_scope('layer1') as scope:
        l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                            strides=[1, 1, 1, 1], padding='SAME'), name='l1a')
        l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1],              # l1 shape=(?, 14, 14, 32)
                            strides=[1, 2, 2, 1], padding='SAME', name='pool')
        l1_bias = tf.Variable(tf.constant(0.05, shape=[32]), name="bias")
        l1 = tf.nn.bias_add(l1, l1_bias)
        l1 = tf.nn.dropout(l1, p_keep_conv, name='dropout')

    with tf.variable_scope('layer2') as scope:
        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                       # l1a shape=(?, 28, 28, 32)
                            strides=[1, 1, 1, 1], padding='SAME'), name='l2a')
        l2 = tf.nn.max_pool(l2a, ksize=[1, 3, 3, 1],              # l1 shape=(?, 14, 14, 32)
                            strides=[1, 2, 2, 1], padding='SAME', name='pool')
        l2_bias = tf.Variable(tf.constant(0.05, shape=[64]), name="bias")
        l2 = tf.nn.bias_add(l2, l2_bias)
        l2 = tf.nn.dropout(l2, p_keep_conv, name='dropout')
#    l2 = tf.nn.lrn(l2, name='l2norm')

#    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                       # l1a shape=(?, 28, 28, 32)
#                        strides=[1, 1, 1, 1], padding='SAME'), name='l3a')
#    l3 = tf.nn.max_pool(l2a, ksize=[1, 8, 8, 1],              # l1 shape=(?, 14, 14, 32)
#                        strides=[1, 2, 2, 1], padding='SAME', name='l3pool')
#    l3 = tf.nn.dropout(l2, p_keep_conv, name='l3dropout')

    with tf.variable_scope('layerfc') as scope:
        l4 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]], name='l4')    # reshape to (?, 14x14x32)

        l_fc = tf.nn.relu(tf.matmul(l4, w_fc), name='l_fully_connected')
        l_fc = tf.nn.dropout(l_fc, p_keep_hidden, name='dropout')

    pyx = tf.matmul(l_fc, w_o, name='pyx')
    return pyx

with tf.name_scope('cnn') as scope:
    X = tf.placeholder('float', [None, 32, 32, 3], name='input')
    Y = tf.placeholder('float', [None, 10], name='output')

    w = init_weights([3, 3, 3, 32], '1')
    w2 = init_weights([3, 3, 32, 64], '2')
    w_fc = init_weights([64 * 8 * 8, 384], 'fc')
    w_o = init_weights([384, 10], 'output')

    p_keep_conv = tf.placeholder('float', name='keep_chance')
    p_keep_hidden = tf.placeholder('float', name='keep_chance')
    py_x = model(X, w, w2, w_fc, w_o, p_keep_conv, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y, name='softmax'), name='cost')
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9, name='optimizer').minimize(cost)
    predict_op = tf.argmax(py_x, 1, name='predict_op')

    te_x, te_y = getTestData()

    # Launch the graph in a session
    with tf.Session() as sess:

        saver = tf.train.Saver()
        if (sys.argv[1] == 'restore' or sys.argv[1] == 'patches' or sys.argv[1] == 'continue') and os.path.exists('mlp/checkpoint'):
            saver.restore(sess, 'mlp/session.ckpt')
        else:
            tf.global_variables_initializer().run()

        if sys.argv[1] == 'patches':
            test_indices = np.arange(len(te_x)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
        else:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            reshaped_x,reshaped_y = [],[]
            accuracies = []
            for i in range(15):
                for b in range(1,total_batches+1):
                    reshaped_x, reshaped_y = getData(b)
                    training_batch = zip(range(0, len(reshaped_x), batch_size), range(batch_size, len(reshaped_x)+1, batch_size))
                    for start, end in training_batch:
                        sess.run(train_op, feed_dict={X: reshaped_x[start:end], Y: reshaped_y[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.8})

                test_indices = np.arange(len(te_x)) # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:test_size]
                accuracy = np.mean(np.argmax(te_y[test_indices], axis=1) == sess.run(predict_op,
                    feed_dict={
                        X: te_x[test_indices],
                        p_keep_conv: 1.0,
                        p_keep_hidden: 1.0
                    })
                )
                print(i, accuracy)
                accuracies.append(accuracy)
                if sys.argv[1] == 'save' or sys.argv[1] == 'continue':
                    saver.save(sess, 'mlp/session.ckpt')
            plt.figure()
            plt.plot(range(1,16), accuracies)
            plt.title("accuracy on epochs")
            plt.show()
