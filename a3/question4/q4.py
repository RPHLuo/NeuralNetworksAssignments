from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class FeedForwardNetwork:

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=1))

    def model(self, X, w_h1, w_o):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1))
        return tf.matmul(self.h1, self.w_o)

    def train(trX, trY):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(1000):
                for start, end in zip(range(0, 90, 10), range(10, 100, 10)):
                    sess.run(self.train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i, sess.run(tf.losses.mean_squared_error(self.teY, sess.run(self.predict_op, feed_dict={X: self.teX}))))

    def __init__(self, size):
        self.X = tf.placeholder("float", [None, 2])
        self.Y = tf.placeholder("float", [None, 1])

        self.size_h1 = tf.constant(size, dtype=tf.int32)
        self.size_h1 = tf.constant(size, dtype=tf.int32)

        self.w_h1 = self.init_weights([2, self.size_h1])
        self.w_h2 = self.init_weights([2, self.size_h1])
        self.w_o = self.init_weights([self.size_h1, 1])
        self.py_x = model(self.X, self.w_h1, self.w_o)

        self.learningRate = 0.02
        self.tolerance = 0.02
        self.cost = tf.losses.mean_squared_error(self.py_x, self.Y)
        self.train_op = tf.train.MomentumOptimizer(self.learningRate,0.9).minimize(cost)
        self.predict_op = py_x

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
#k-folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(np.array(lfw_people.data).shape)
print(np.array(X_train_pca).shape)
