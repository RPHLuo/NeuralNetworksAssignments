import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

total_batches = 5
batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l1 = tf.nn.lrn(l1, 4)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 3, 3, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

#    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
#    conv2 = tf.nn.relu(conv2)
#    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

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

with tf.name_scope('cnn') as scope:
    X = tf.placeholder("float", [None, 32, 32, 3], name='input')
    Y = tf.placeholder("float", [None, 10], name='output')

    w = init_weights([5, 5, 3, 64])       # 3x3x1 conv, 32 outputs
    w2 = init_weights([5, 5, 64, 64])       # 3x3x1 conv, 32 outputs
    w_fc = init_weights([64 * 8 * 8, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
    w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

    p_keep_conv = tf.placeholder("float", name='keep_conv')
    p_keep_hidden = tf.placeholder("float", name='keep_hidden')
    py_x = model(X, w, w2, w_fc, w_o, p_keep_conv, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    te_x, te_y = getTestData()

    # Launch the graph in a session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        tf.global_variables_initializer().run()
        reshaped_x,reshaped_y = [],[]
        for i in range(15):
            for b in range(1,total_batches+1):
                reshaped_x, reshaped_y = getData(b)
                training_batch = zip(range(0, len(reshaped_x), batch_size), range(batch_size, len(reshaped_x)+1, batch_size))
                for start, end in training_batch:
                    sess.run(train_op, feed_dict={X: reshaped_x[start:end], Y: reshaped_y[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})

            test_indices = np.arange(len(te_x)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            print(i, np.mean(
                np.argmax(te_y[test_indices], axis=1) == sess.run(predict_op,
                feed_dict={
                    X: te_x[test_indices],
                    p_keep_conv: 1.0,
                    p_keep_hidden: 1.0
                })))
