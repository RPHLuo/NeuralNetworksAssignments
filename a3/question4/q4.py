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
        h2 = tf.nn.leaky_relu(tf.matmul(h1, self.w_h2))
        return tf.matmul(h2, self.w_o)

    def train(self, trX, trY, teX, teY):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(10):
                for start, end in zip(range(0, len(trX), 100), range(100, len(trX)+1, 100)):
                    sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
                print(sess.run(self.predict_op, feed_dict={self.X: teX}))
                print(np.argmax(teY, axis=1))
                print(i, np.mean(np.argmax(teY, axis=1) == sess.run(self.predict_op, feed_dict={self.X: teX})))

    def __init__(self, in_size, h1_size, h2_size, out_size):
        self.X = tf.placeholder("float", [None, in_size])
        self.Y = tf.placeholder("float", [None, out_size])

        self.size_h1 = tf.constant(h1_size, dtype=tf.int32)
        self.size_h2 = tf.constant(h2_size, dtype=tf.int32)

        self.w_h1 = self.init_weights([in_size, self.size_h1])
        self.w_h2 = self.init_weights([self.size_h1, self.size_h2])
        self.w_o = self.init_weights([self.size_h2, out_size])

        self.py_x = self.model(self.X, self.w_h1, self.w_h2, self.w_o)

        self.learningRate = 0.02
        self.tolerance = 0.02
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        self.train_op = tf.train.GradientDescentOptimizer(0.05).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

def input_data(raw_data=True):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]
    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    # One-hot the label data
    labels = np.zeros(shape=(y.shape[0], y.max() + 1))
    for i, x in enumerate(y):
        labels[i][x] = 1
    if raw_data:
        X = X / X.max() # Regularize the input data
    else:
        # Compute principle components of input data
        X = PCA(n_components=150, svd_solver='randomized', whiten=True).fit_transform(X)
    return X, labels

X_pca,labels_pca = input_data(raw_data=False)
X_raw,labels_raw = input_data(raw_data=True)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, labels_raw, test_size=0.25, random_state=42)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, labels_pca, test_size=0.25, random_state=42)
raw_data_nn = FeedForwardNetwork(1850, 600, 200, 7)
opt_data_nn = FeedForwardNetwork(150, 100, 50, 7)
opt_data_nn.train(X_train_pca, y_train_pca, X_test_pca, y_test_pca)
raw_data_nn.train(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
