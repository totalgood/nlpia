
# coding: utf-8

# In[7]:


import keras
import numpy as np


# In[37]:


# Our examples of exclusive OR.
# x_train is sample data
# y_train the expected outcome for example
x_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
y_train = np.array([[0],
                    [1],
                    [1],
                    [0]])


# In[45]:


# The base Keras model class

from keras.models import Sequential

# The basic layer of the network
# Dense is a fully-connected set of neurons

from keras.layers import Dense, Activation

# Get stochastic gradient descent, though there are others
from keras.optimizers import SGD


# In[67]:


model = Sequential()

# Add a fully connected hidden layer with 10 neurons
# The input shape is the shape of an individual sample vector
# This is only necessary in the first layer, any additional
# layers will calculate the shape automatically by the definition
# of the model up to that point

num_neurons = 10
model.add(Dense(num_neurons, input_dim=2))
model.add(Activation('tanh'))

# The output layer one neuron to output 0 or 1 
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())


# In[68]:


print(model.predict(x_train))


# In[69]:


sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[70]:


# Here is where we train the model
model.fit(x_train, y_train,
          epochs=100)


# In[62]:


# And then we save the structure and learned weights for later use
model_structure = model.to_json()
with open("basic_model.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("basic_weights.h5")
print('Model saved.')


# In[63]:


print(model.predict_classes(x_train))
print(model.predict(x_train))

