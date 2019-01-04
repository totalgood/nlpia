# ch05m01_your_first_neuron.py
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential

# OR logic gate inputs and output.
# x_train is sample data (input features)
# y_train the expected outcome for example
x_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
y_train = np.array([[0],
                    [1],
                    [1],
                    [1]])
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))
model.compile('SGD', 'mse')
model.fit(x_train, y_train)
# Get stochastic gradient descent, though there are others
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))
model.compile('sgd', 'mse')
model.fit(x=np.array(x_train), y=np.array(y_train), epochs=5000, batch_size=4)
