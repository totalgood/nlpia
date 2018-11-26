
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Model
from ch10 import construct_seq2seq_model


# In[ ]:


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
num_samples = 10000
data_path = '../data/dialog.txt'  # preprocessed CMU movie dialogue samples


# In[ ]:


try:
    import cPickle as pickle
except ImportError:
    import pickle

from io import open

with open("../data/characters_stats.pkl", "rb") as filehandler:
    input_characters, target_characters, input_token_index, target_token_index = pickle.load(filehandler)

with open("../data/encoder_decoder_stats.pkl", "rb") as filehandler:
    num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length = pickle.load(filehandler)


# In[ ]:


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path).read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[ ]:


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# In[ ]:


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# In[ ]:


model = construct_seq2seq_model(num_encoder_tokens, num_decoder_tokens)


# In[ ]:


# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

