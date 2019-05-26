
# coding: utf-8

# In[ ]:


import os
import re
import tarfile

from pugnlp.futil import path_status, find_files

from nlpia.web import requests_get


# In[ ]:


# From the nlpia package for downloading data too big for the repo

BIG_URLS = {
    'w2v': (
        'https://www.dropbox.com/s/965dir4dje0hfi4/GoogleNews-vectors-negative300.bin.gz?dl=1',
        1647046227,
    ),
    'slang': (
        'https://www.dropbox.com/s/43c22018fbfzypd/slang.csv.gz?dl=1',
        117633024,
    ),
    'tweets': (
        'https://www.dropbox.com/s/5gpb43c494mc8p0/tweets.csv.gz?dl=1',
        311725313,
    ),
    'lsa_tweets': (
        'https://www.dropbox.com/s/rpjt0d060t4n1mr/lsa_tweets_5589798_2003588x200.tar.gz?dl=1',
        3112841563,  # 3112841312,
    ),
    'imdb': (
        'https://www.dropbox.com/s/yviic64qv84x73j/aclImdb_v1.tar.gz?dl=1',
        3112841563,  # 3112841312,
    ),
}


# In[ ]:


# These functions are part of the nlpia package which can be pip installed and run from there.
def dropbox_basename(url):
    filename = os.path.basename(url)
    match = re.findall(r'\?dl=[0-9]$', filename)
    if match:
        return filename[:-len(match[0])]
    return filename


def download_file(url, data_path='.', filename=None, size=None, chunk_size=4096, verbose=True):
    """Uses stream=True and a reasonable chunk size to be able to download large (GB) files over https"""
    if filename is None:
        filename = dropbox_basename(url)
    file_path = os.path.join(data_path, filename)
    if url.endswith('?dl=0'):
        url = url[:-1] + '1'  # noninteractive download
    if verbose:
        tqdm_prog = tqdm
        print('requesting URL: {}'.format(url))
    else:
        tqdm_prog = no_tqdm
    r = requests_get(url, stream=True, allow_redirects=True, timeout=5)
    size = r.headers.get('Content-Length', None) if size is None else size
    print('remote size: {}'.format(size))

    stat = path_status(file_path)
    print('local size: {}'.format(stat.get('size', None)))
    if stat['type'] == 'file' and stat['size'] == size:  # TODO: check md5 or get the right size of remote file
        r.close()
        return file_path

    print('Downloading to {}'.format(file_path))

    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

    r.close()
    return file_path


def untar(fname):
    if fname.endswith("tar.gz"):
        with tarfile.open(fname) as tf:
            tf.extractall()
    else:
        print("Not a tar.gz file: {}".format(fname))


# In[ ]:


#  UNCOMMENT these 2 lines if you haven't already download the word2vec model and the imdb dataset
# download_file(BIG_URLS['w2v'][0])
# untar(download_file(BIG_URLS['imdb'][0]))


# In[ ]:


maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

num_neurons = 50

print('Build model...')
model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


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


# In[ ]:


from nltk.tokenize import TreebankWordTokenizer
from gensim.models import KeyedVectors
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


# In[ ]:


def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


# In[ ]:


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


# In[ ]:


import numpy as np

dataset = pre_process_data('./aclImdb_v1/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 400
batch_size = 32         # How many samples to show the net before backpropogating the error and updating the weights
embedding_dims = 300    # Length of the token vectors we will create for passing into the Convnet
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

num_neurons = 50

print('Build model...')
model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("lstm_model1.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("lstm_weights1.h5")
print('Model saved.')


# In[ ]:


from keras.models import model_from_json
with open("lstm_model1.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

model.load_weights('lstm_weights1.h5')


# In[ ]:


sample_1 = "I'm hate that the dismal weather that had me down for so long, when will it break! Ugh, when does happiness return?  The sun is blinding and the puffy clouds are too thin.  I can't wait for the weekend."

# We pass a dummy value in the first element of the tuple just because our helper expects it from the way processed the initial data.  That value won't ever see the network, so it can be whatever.
vec_list = tokenize_and_vectorize([(1, sample_1)])

# Tokenize returns a list of the data (length 1 here)
test_vec_list = pad_trunc(vec_list, maxlen)

test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))

print("Sample's sentiment, 1 - pos, 2 - neg : {}".format(model.predict_classes(test_vec)))
print("Raw output of sigmoid function: {}".format(model.predict(test_vec)))


# In[ ]:


def test_len(data, maxlen):
    total_len = truncated = exact = padded = 0
    for sample in data:
        total_len += len(sample)
        if len(sample) > maxlen:
            truncated += 1
        elif len(sample) < maxlen:
            padded += 1
        else:
            exact += 1
    print('Padded: {}'.format(padded))
    print('Equal: {}'.format(exact))
    print('Truncated: {}'.format(truncated))
    print('Avg length: {}'.format(total_len / len(data)))


dataset = pre_process_data('./aclImdb_v1/train')
vectorized_data = tokenize_and_vectorize(dataset)
test_len(vectorized_data, 400)


# In[ ]:


import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM


maxlen = 200
batch_size = 32         # How many samples to show the net before backpropagating the error and updating the weights
embedding_dims = 300    # Length of the token vectors we will create for passing into the Convnet

epochs = 2

dataset = pre_process_data('./aclImdb_v1/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

num_neurons = 50

print('Build model...')
model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("lstm_model7.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("lstm_weights7.h5")
print('Model saved.')


# In[ ]:


dataset = pre_process_data('./aclImdb_v1/train')
expected = collect_expected(dataset)


# In[ ]:


def avg_len(data):
    total_len = 0
    for sample in data:
        total_len += len(sample[1])
    print(total_len / len(data))


print(avg_len(dataset))


# In[ ]:


def clean_data(data):
    """ Shift to lower case, replace unknowns with UNK, and listify """
    new_data = []
    VALID = 'abcdefghijklmnopqrstuvwxyz123456789"\'?!.,:; '
    for sample in data:
        new_sample = []
        for char in sample[1].lower():  # Just grab the string, not the label
            if char in VALID:
                new_sample.append(char)
            else:
                new_sample.append('UNK')

        new_data.append(new_sample)
    return new_data


listified_data = clean_data(dataset)


# In[ ]:


def char_pad_trunc(data, maxlen):
    """ We truncate to maxlen or add in PAD tokens """
    new_dataset = []
    for sample in data:
        if len(sample) > maxlen:
            new_data = sample[:maxlen]
        elif len(sample) < maxlen:
            pads = maxlen - len(sample)
            new_data = sample + ['PAD'] * pads
        else:
            new_data = sample
        new_dataset.append(new_data)
    return new_dataset


maxlen = 1500


# In[ ]:


def create_dicts(data):
    """ Modified from Keras LSTM example"""
    chars = set()
    for sample in data:
        chars.update(set(sample))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char


# In[ ]:


import numpy as np


def onehot_encode(dataset, char_indices, maxlen):
    """
    One hot encode the tokens

    Args:
        dataset  list of lists of tokens
        char_indices  dictionary of {key=character, value=index to use encoding vector}
        maxlen  int  Length of each sample
    Return:
        np array of shape (samples, tokens, encoding length)
    """
    X = np.zeros((len(dataset), maxlen, len(char_indices.keys())))
    for i, sentence in enumerate(dataset):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
    return X


# In[ ]:


dataset = pre_process_data('./aclImdb_v1/train')
expected = collect_expected(dataset)
listified_data = clean_data(dataset)

maxlen = 1500
common_length_data = char_pad_trunc(listified_data, maxlen)

char_indices, indices_char = create_dicts(common_length_data)
encoded_data = onehot_encode(common_length_data, char_indices, maxlen)


# In[ ]:


split_point = int(len(encoded_data) * .8)

x_train = encoded_data[:split_point]
y_train = expected[:split_point]
x_test = encoded_data[split_point:]
y_test = expected[split_point:]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM


num_neurons = 40

print('Build model...')
model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, len(char_indices.keys()))))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


# In[ ]:


batch_size = 32
epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("char_lstm_model3.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("char_lstm_weights3.h5")
print('Model saved.')


# In[ ]:


from nltk.corpus import gutenberg

print(gutenberg.fileids())


# In[ ]:


text = ''
for txt in gutenberg.fileids():
    if 'shakespeare' in txt:
        text += gutenberg.raw(txt).lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[ ]:


print(text[:500])


# In[ ]:


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


# In[ ]:


print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print(model.summary())


# In[ ]:


epochs = 6
batch_size = 128

model_structure = model.to_json()
with open("shakes_lstm_model.json", "w") as json_file:
    json_file.write(model_structure)

for i in range(5):
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs)

    model.save_weights("shakes_lstm_weights_{}.h5".format(i + 1))
    print('Model saved.')


# In[ ]:


# NOT IN CHAPTER, Just to reproduce output

from keras.models import model_from_json

with open('shakes_lstm_model.json', 'r') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights('shakes_lstm_weights_4.h5')


# In[ ]:


import random


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:


import sys

start_index = random.randint(0, len(text) - maxlen - 1)

for diversity in [0.2, 0.5, 1.0]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


# In[ ]:


from keras.models import Sequential
from keras.layers import GRU

model = Sequential()
model.add(GRU(num_neurons, return_sequences=True, input_shape=X[0].shape))


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(num_neurons, return_sequences=True, input_shape=X[0].shape))
model.add(LSTM(num_neurons_2, return_sequences=True))

