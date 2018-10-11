from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector


# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# decode a one hot encoded string


def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM


def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, cardinality)
    y = one_hot_encode(sequence_out, cardinality)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(5000):
    # generate new random sequence
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, verbose=2)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# spot check some examples
for _ in range(10):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
