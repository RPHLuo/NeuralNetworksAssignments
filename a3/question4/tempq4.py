#!/usr/bin/env python
# coding: utf-8

# ## This question has been inspired from a couple of online resources. For information please email us at nazalkassm@cmail.carleton.ca

# In[13]:


from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA # Allows us to use PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


# ## Taking care of inputing the data

# In[14]:


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
        X = PCA(n_components=70, svd_solver='randomized', whiten=True).fit_transform(X)
        

    return X, labels


# In[26]:


X_pca,labels_pca = input_data(raw_data=False)
X_raw,labels_raw = input_data(raw_data=True)


# ## Playing with the data

# In[16]:


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# In[17]:


image1= lfw_people.images[0]
plt.imshow(image1)
plt.show()


# In[18]:


image2 = lfw_people.data[0,:]


# In[19]:


print(image2.shape)
print(image1.shape)


# In[20]:


image3 = image2.reshape(50,37)


# In[24]:


plt.imshow(image3)
plt.show()


# ## Defining our Neural Network

# In[28]:


class FeedForwardNetwork(object):
    
    def __init__(self, hidden_layer1, hidden_layer2, output_layer, input_size=1850, learning_rate=0.002):

        def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=1))
        
        self.X = tf.placeholder("float", [None, input_size])
        slef.Y = tf.placeholder("float", [None, output_layer])

        self.size_h1 = tf.constant(size, dtype=tf.int32)
        self.size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)

        self.w_h1 = self.__init_weights(input_size, hidden_layer1)
        self.w_h2 = self.__init_weights(hidden_layer1, hidden_layer2)
        self.w_o = self.__init_weights(hidden_layer2, output_layer)
        
        self.py_x = self.__model()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)
        
        self.accuracies = []
        self.mean_accuracy = None
        
    def __model(self):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1), alpha=0.2)
        h2 = tf.nn.leaky_relu(tf.matmul(h1, self.w_h2), alpha=0.2)
        return tf.matmul(h2, self.w_o)


# In[ ]:




