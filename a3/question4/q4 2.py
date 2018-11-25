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


# In[22]:


X,labels = input_data(raw_data=False)
X,labels = input_data(raw_data=True)


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

# In[25]:


class FacialRecognitionNetwork(object):
    def __init__(self, hidden_layer1, hidden_layer2, output_layer, input_size=1850, learning_rate=0.002):
        
        
       


# In[ ]:




