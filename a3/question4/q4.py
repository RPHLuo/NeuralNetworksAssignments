from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class FeedForwardNetwork(object):
    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=1))

    def model(self, X, w_h1, w_h2, w_o):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1))
        h2 = tf.nn.leaky_relu(tf.matmul(h1, self.w_h2))
        return tf.matmul(h2, self.w_o)

    def train(self, trX, trY, teX, teY):
        results = []
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(10):
                for start, end in zip(range(0, len(trX), 100), range(100, len(trX)+1, 100)):
                    sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
                remainder = len(trX) % 100
                sess.run(self.train_op, feed_dict={self.X: trX[(len(trX)-remainder):], self.Y: trY[(len(trX)-remainder):]})
                accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(self.predict_op, feed_dict={self.X: teX}))
                results.append(accuracy)
                print('epoch: ',i, ' accuracy: ', accuracy)
        return results

    def __init__(self, in_size, h1_size, h2_size, out_size):
        self.X = tf.placeholder("float", [None, in_size])
        self.Y = tf.placeholder("float", [None, out_size])

        self.size_h1 = tf.constant(h1_size, dtype=tf.int32)
        self.size_h2 = tf.constant(h2_size, dtype=tf.int32)

        self.w_h1 = self.init_weights([in_size, self.size_h1])
        self.w_h2 = self.init_weights([self.size_h1, self.size_h2])
        self.w_o = self.init_weights([self.size_h2, out_size])

        self.py_x = self.model(self.X, self.w_h1, self.w_h2, self.w_o)

        self.learning_rate = 0.02
        self.tolerance = 0.02
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

def input_data(raw_data=True):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    labels = np.zeros(shape=(y.shape[0], y.max() + 1))
    for i, x in enumerate(y):
        labels[i][x] = 1
    if raw_data:
        X /= X.max()
    else:
        X = PCA(n_components=75, svd_solver='randomized', whiten=True).fit_transform(X)
    return X, labels

X_pca,labels_pca = input_data(raw_data=False)
X_raw,labels_raw = input_data(raw_data=True)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, labels_raw, test_size=0.1, random_state=1)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, labels_pca, test_size=0.1, random_state=1)

raw_data_nn = FeedForwardNetwork(len(X_train_raw[0]), 200, 50, len(y_train_raw[0]))
print('input layer size: ', len(X_train_raw[0]), 'h1 size:', 200, 'h2 size:', 50, 'output size: ', len(y_train_raw[0]))
print('raw data trained')
raw_results = raw_data_nn.train(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
plt.plot(np.arange(1,11),raw_results)

opt_data_nn = FeedForwardNetwork(len(X_train_pca[0]), 200, 50, len(y_train_pca[0]))
print('input layer size: ', len(X_train_pca[0]), 'h1 size:', 200, 'h2 size:', 50, 'output size: ', len(y_train_pca[0]))
print('optimized pca trained')
opt_results = opt_data_nn.train(X_train_pca, y_train_pca, X_test_pca, y_test_pca)
plt.plot(np.arange(1,11),opt_results)
plt.show()
