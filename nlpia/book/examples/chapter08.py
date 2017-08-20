
# coding: utf-8

# In[1]:


import glob
import os

from random import shuffle

def pre_process_data(filepath):
    """
    This is dependent on your training data source but we will try to generalize it as best as possible.
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    
    pos_label = 1
    neg_label = 0
    
    dataset = []
    
    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))
            
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))
    
    shuffle(dataset)
    
    return dataset


# In[2]:


from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])

            except KeyError:
                pass  # No matching token in the Google w2v vocab
            
        vectorized_data.append(sample_vecs)

    return vectorized_data


# In[3]:


word_vectors["dog"]


# In[4]:


def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


# In[5]:


dataset = pre_process_data('./aclimdb/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)


# In[6]:


split_point = int(len(vectorized_data)*.8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


# In[7]:


maxlen = 400
batch_size = 32         # How many samples to show the net before backpropogating the error and updating the weights
embedding_dims = 300    # Length of the token vectors we will create for passing into the Convnet

epochs = 2


# In[8]:


def pad_trunc(data, maxlen):
    """ For a given dataset pad with zero vectors or truncate to maxlen """
    new_data = []

    # Create a vector of 0's the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
 
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


# In[9]:


import numpy as np

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN

num_neurons = 50

print('Build model...')
model = Sequential()

model.add(SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy',  metrics=['accuracy'])
print(model.summary())


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("simplernn_model1.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("simplernn_weights1.h5")
print('Model saved.')


# In[ ]:


from keras.models import model_from_json
with open("simplernn_model1.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

model.load_weights('simplernn_weights1.h5')


# In[ ]:


sample_1 = "I'm hate that the dismal weather that had me down for so long, when will it break! Ugh, when does happiness return?  The sun is blinding and the puffy clouds are too thin.  I can't wait for the weekend."


# In[ ]:


# We pass a dummy value in the first element of the tuple just because our helper expects it from the way processed the initial data.  That value won't ever see the network, so it can be whatever.
vec_list = tokenize_and_vectorize([(1, sample_1)])

# Tokenize returns a list of the data (length 1 here)
test_vec_list = pad_trunc(vec_list, maxlen)

test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))


# In[ ]:


model.predict_classes(test_vec)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN

num_neurons = 100

print('Build model...')
model = Sequential()

model.add(SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy',  metrics=['accuracy'])
print(model.summary())


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("simplernn_model2.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("simplernn_weights2.h5")
print('Model saved.')

