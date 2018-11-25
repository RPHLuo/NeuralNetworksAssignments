from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class FeedForwardNetwork:

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=1))

    def model(self, X, w_h1, w_h2, w_o):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1))
        h2 = tf.nn.leaky_relu(tf.matmul(self.w_h1, self.w_h2))
        return tf.matmul(self.w_h2, self.w_o)

    def train(self, trX, trY, teX, teY):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(1000):
                sess.run(self.train_op, feed_dict={self.X: trX, self.Y: trY})
            print(i, sess.run(tf.losses.mean_squared_error(teY, sess.run(self.predict_op, feed_dict={X: teX}))))

    def __init__(self, size, h1_size, h2_size):
        self.X = tf.placeholder("float", [None, size])
        self.Y = tf.placeholder("float", [None, 7])

        self.input_size = tf.constant(size, dtype=tf.int32)
        self.size_h1 = tf.constant(h1_size, dtype=tf.int32)
        self.size_h2 = tf.constant(h2_size, dtype=tf.int32)

        self.w_h1 = self.init_weights([self.input_size, self.size_h1])
        self.w_h2 = self.init_weights([self.size_h1, self.size_h2])
        self.w_o = self.init_weights([self.size_h2, 7])

        self.py_x = self.model(self.X, self.w_h1, self.w_h2, self.w_o)

        self.learningRate = 0.02
        self.tolerance = 0.02
        self.cost = tf.losses.mean_squared_error(self.py_x, self.Y)
        self.train_op = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.cost)
        self.predict_op = self.py_x

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
print(y_train)
print(np.array(X_train_pca).shape)
print(np.array(X_test_pca).shape)
print(np.array(X_train_pca).shape)
raw_data_nn = FeedForwardNetwork(1850, 1200, 600)
opt_data_nn = FeedForwardNetwork(150, 100, 50)
opt_data_nn.train(X_train_pca, y_train, X_test_pca, y_train)
raw_data_nn.train(X_train, y_train, X_test, y_test)
