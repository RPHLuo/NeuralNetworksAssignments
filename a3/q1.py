import sklearn as sk
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Hopfield_Network:
    def __init__(self, num_units, scope='hopfield_network'):
        # pylint: disable=E1129
        with tf.variable_scope(scope):
            self._weights = tf.get_variable('weights',
                                            shape=(num_units, num_units),
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
            self._thresholds = tf.get_variable('thresholds',
                                               shape=(num_units,),
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

    @property
    def weights(self):
        return self._weights

    @property
    def thresholds(self):
        return self._thresholds

    def update(samples, weights):
        return

    def hebbian_update(layer):
        converged = False
        while(not converged):
            converged = True
            update_vector = tf.zeros(len(layer))
            for i in range(0, len(layer)):
                if layer[i] != update_vector[i]:
                    update_vector[i] = hebb_neuron(layer, i)
                    converged = False
        return update_vector

    def hebb_neuron(neurons, i):
        total = 0
        for n in range(0, len(neurons)):
            total += neurons[n] * neurons[i]
        total /= len(neurons)
        return total

    def storkey_learning():
        return

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


ones = trY[:, 1]
fives = trY[:, 5]
modified_trX = [trX[i] for i in range(0,len(trX)) if ones[i] != 0 or fives[i] != 0]
print(np.array(modified_trX).shape)
ones = np.matrix(ones).T
fives = np.matrix(fives).T
trY = np.array(np.concatenate((ones,fives), axis=1))
modified_trY = [trY[i] for i in range(0,len(trY)) if trY[i][0] != 0 or trY[i][1] != 0]
print(np.array(modified_trY).shape)

input = np.concatenate((modified_trX, modified_trY), axis=1)
print(input.shape)


# 784 inputs + 2 outputs
hopfield = Hopfield_Network(786)

layer = tf.placeholder(tf.float32, shape=(786,))
update = hopfield.hebbian_update
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for image in input:
        sess.run(update, feed_dict={layer: image})
