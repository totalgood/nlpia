r""" Build character sequence-to-sequence training set
>>> import os
>>> from nlpia.loaders import get_data
>>> df = get_data(os.path.join('..', 'book', 'data', 'dialog.txt'))
>>> df.columns = 'statement reply'.split()
>>> df = df.fillna(' ')
>>> input_texts, target_texts = [], []  # <1>
>>> input_vocabulary = set()  # <2>
>>> output_vocabulary = set()
>>> start_token = '\t'  # <3>
>>> stop_token = '\n'
>>> max_training_samples = min(25000, len(df) - 1)  # <4>

>>> for input_text, target_text in zip(df.statement, df.reply):
...     target_text = start_token + target_text \
...         + stop_token  # <5>
...     input_texts.append(input_text)
...     target_texts.append(target_text)
...     for char in input_text:  # <6>
...         if char not in input_vocabulary:
...             input_vocabulary.add(char)
...     for char in target_text:
...         if char not in output_vocabulary:
...             output_vocabulary.add(char)
"""
import os
from nlpia.loaders import get_data, DATA_PATH
df = get_data(os.path.join(DATA_PATH, '..', 'book', 'data', 'dialog.txt'))
df.columns = 'statement reply'.split()
df = df.fillna(' ')
input_texts, target_texts = [], []  # <1>
input_vocabulary = set()  # <2>
output_vocabulary = set()
start_token = '\t'  # <3>
stop_token = '\n'
max_training_samples = min(25000, len(df) - 1)  # <4>

for input_text, target_text in zip(df.statement, df.reply):
    target_text = start_token + target_text \
        + stop_token  # <5>
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:  # <6>
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)
# <1> The arrays hold the input and target text read from the corpus file.
# <2> The sets hold the seen characters in the input and target text.
# <3> The target sequence is annotated with a start (first) and stop (last) token; the characters representing the tokens are defined here. These tokens can't be part of the normal sequence text and should be uniquely used as start and stop tokens.
# <4> `max_training_samples` defines how many lines are used for the training. It is the lower number of either a user-defined maximum or the total number of lines loaded from the file.
# <5> The `target_text` needs to be wrapped with the start and stop tokens.
# <6> Compile the vocabulary -- set of the unique characters seen in the input_texts


r""" Character sequence-to-sequence model parameters
>>> input_vocabulary = sorted(input_vocabulary)  # <1>
>>> output_vocabulary = sorted(output_vocabulary)

>>> input_vocab_size = len(input_vocabulary)  # <2>
>>> output_vocab_size = len(output_vocabulary)
>>> max_encoder_seq_length = max(
...     [len(txt) for txt in input_texts])  # <3>
>>> max_decoder_seq_length = max(
...     [len(txt) for txt in target_texts])

>>> input_token_index = dict([(char, i) for i, char in
...     enumerate(input_vocabulary)])  # <4>
>>> target_token_index = dict(
...     [(char, i) for i, char in enumerate(output_vocabulary)])
>>> reverse_input_char_index = dict((i, char) for char, i in
...     input_token_index.items())  # <5>
>>> reverse_target_char_index = dict((i, char) for char, i in
...     target_token_index.items())
"""
input_vocabulary = sorted(input_vocabulary)  # <1>
output_vocabulary = sorted(output_vocabulary)

input_vocab_size = len(input_vocabulary)  # <2>
output_vocab_size = len(output_vocabulary)
max_encoder_seq_length = max(
    [len(txt) for txt in input_texts])  # <3>
max_decoder_seq_length = max(
    [len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_vocabulary)])  # <4>
target_token_index = dict(
    [(char, i) for i, char in enumerate(output_vocabulary)])
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())  # <5>
# reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
# <1> You convert the character sets into sorted lists of characters, which you then use to generate the dictionary.
# <2> For the input and target data, you determine the maximum number of unique characters, which you use to build the one-hot matrices.
# <3> For the input and target data, you also determine the maximum number of sequence tokens.
# <4> Loop over the `input_characetrs` and `output_vocabulary` to create the lookup dictionaries, which you use to generate the one-hot vectors.
# <5> Loop over the newly created dictionaries to create the reverse lookups.


r""" Construct character sequence encoder-decoder training set

>>> import numpy as np  # <1>

>>> encoder_input_data = np.zeros((len(input_texts),
...     max_encoder_seq_length, input_vocab_size),
...     dtype='float32')  # <2>
>>> decoder_input_data = np.zeros((len(input_texts),
...     max_decoder_seq_length, output_vocab_size),
...     dtype='float32')
>>> decoder_target_data = np.zeros((len(input_texts),
...     max_decoder_seq_length, output_vocab_size),
...     dtype='float32')

>>> for i, (input_text, target_text) in enumerate(
...             zip(input_texts, target_texts)):  # <3>
...     for t, char in enumerate(input_text):  # <4>
...         encoder_input_data[
...             i, t, input_token_index[char]] = 1.  # <5>
...     for t, char in enumerate(target_text):  # <6>
...         decoder_input_data[
...             i, t, target_token_index[char]] = 1.
...         if t > 0:
...             decoder_target_data[i, t - 1, target_token_index[char]] = 1
"""
import numpy as np  # <1> # noqa

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, input_vocab_size),
    dtype='float32')  # <2>
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, output_vocab_size),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, output_vocab_size),
    dtype='float32')

for i, (input_text, target_text) in enumerate(
        zip(input_texts, target_texts)):  # <3>
    for t, char in enumerate(input_text):  # <4>
        encoder_input_data[
            i, t, input_token_index[char]] = 1.  # <5>
    for t, char in enumerate(target_text):  # <6>
        decoder_input_data[
            i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1
# <1> You use numpy for the matrix manipulations.
# <2> The training tensors are initialized as zero tensors with the shape of number of samples (this number should be equal for the input and target samples) times the maximum number of sequence tokens times the number of possible characters.
# <3> Loop over the training samples; input and target texts need to match.
# <4> Loop over each character of each sample.
# <5> Set the index for the character at each time step to one; all other indices remain at zero. This creates the one-hot encoded representation of the training samples.
# <6> For the training data for the decoder, you create the `decoder_input_data` and `decoder_target_data` (which is one time step behind the _decoder_input_data_).


r"""Construct and train a character sequence encoder-decoder network
>>> from keras.models import Model
>>> from keras.layers import Input, LSTM, Dense

>>> batch_size = 64    # <1>
>>> epochs = 100       # <2>
>>> num_neurons = 256  # <3>

>>> encoder_inputs = Input(shape=(None, input_vocab_size))
>>> encoder = LSTM(num_neurons, return_state=True)
>>> encoder_outputs, state_h, state_c = encoder(encoder_inputs)
>>> encoder_states = [state_h, state_c]

>>> decoder_inputs = Input(shape=(None, output_vocab_size))
>>> decoder_lstm = LSTM(num_neurons, return_sequences=True,
...                     return_state=True)
>>> decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
...     initial_state=encoder_states)
>>> decoder_dense = Dense(output_vocab_size, activation='softmax')
>>> decoder_outputs = decoder_dense(decoder_outputs)
>>> model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

>>> model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
...               metrics=['acc'])
>>> model.fit([encoder_input_data, decoder_input_data],
...     decoder_target_data, batch_size=batch_size, epochs=epochs,
...     validation_split=0.1)  # <4>
"""
from keras.models import Model  # noqa
from keras.layers import Input, LSTM, Dense  # noqa

batch_size = 64    # <1>
epochs = 100       # <2>
num_neurons = 256  # <3>

encoder_inputs = Input(shape=(None, input_vocab_size))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequences=True,
                    return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['acc'])
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data, batch_size=batch_size, epochs=epochs,
          validation_split=0.1)  # <4>
# 57915/57915 [==============================] - 296s 5ms/step - loss: 0.7575 - acc: 0.1210 - val_loss: 0.6521 - val_acc: 0.1517
# Epoch 2/100
# 57915/57915 [==============================] - 283s 5ms/step - loss: 0.5924 - acc: 0.1613 - val_loss: 0.5738 - val_acc: 0.1734
# Epoch 3/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.5401 - acc: 0.1755 - val_loss: 0.5393 - val_acc: 0.1837
# Epoch 4/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.5124 - acc: 0.1829 - val_loss: 0.5193 - val_acc: 0.1889
# Epoch 5/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4950 - acc: 0.1878 - val_loss: 0.5074 - val_acc: 0.1924
# Epoch 6/100
# ...
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4537 - acc: 0.1990 - val_loss: 0.4798 - val_acc: 0.2001
# Epoch 11/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4490 - acc: 0.2002 - val_loss: 0.4768 - val_acc: 0.2006
# ...
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4299 - acc: 0.2056 - val_loss: 0.4700 - val_acc: 0.2030
# Epoch 18/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4276 - acc: 0.2062 - val_loss: 0.4689 - val_acc: 0.2035
# Epoch 19/100
# ...
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4151 - acc: 0.2098 - val_loss: 0.4695 - val_acc: 0.2035
# Epoch 26/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4137 - acc: 0.2102 - val_loss: 0.4697 - val_acc: 0.2037
# ...
# Epoch 32/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4064 - acc: 0.2123 - val_loss: 0.4717 - val_acc: 0.2035
# Epoch 33/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4053 - acc: 0.2127 - val_loss: 0.4729 - val_acc: 0.2032
# ...
# Epoch 69/100                                                                                                                                                                                                                           [1480/1902]
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3830 - acc: 0.2191 - val_loss: 0.4912 - val_acc: 0.2008
# Epoch 70/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3826 - acc: 0.2193 - val_loss: 0.4902 - val_acc: 0.2007
# ...
# Epoch 99/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3738 - acc: 0.2220 - val_loss: 0.5000 - val_acc: 0.1994
# Epoch 100/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3736 - acc: 0.2220 - val_loss: 0.5017 - val_acc: 0.1992

r""" .Construct response generator model
>>> encoder_model = Model(encoder_inputs, encoder_states)
>>> thought_input = [
...     Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
>>> decoder_outputs, state_h, state_c = decoder_lstm(
...     decoder_inputs, initial_state=thought_input)
>>> decoder_states = [state_h, state_c]
>>> decoder_outputs = decoder_dense(decoder_outputs)

>>> decoder_model = Model(
...     inputs=[decoder_inputs] + thought_input,
...     output=[decoder_outputs] + decoder_states)
"""
encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [
    Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=thought_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    inputs=[decoder_inputs] + thought_input,
    output=[decoder_outputs] + decoder_states)

r"""
>>> def decode_sequence(input_seq):
...     thought = encoder_model.predict(input_seq)  # <1>

...     target_seq = np.zeros((1, 1, output_vocab_size))  # <2>
...     target_seq[0, 0, target_token_index[stop_token]
...         ] = 1.  # <3>
...     stop_condition = False
...     generated_sequence = ''

...     while not stop_condition:
...         output_tokens, h, c = decoder_model.predict(
...             [target_seq] + thought) # <4>

...         generated_token_idx = np.argmax(output_tokens[0, -1, :])
...         generated_char = reverse_target_char_index[generated_token_idx]
...         generated_sequence += generated_char
...         if (generated_char == stop_token or
...                 len(generated_sequence) > max_decoder_seq_length
...                 ):  # <5>
...             stop_condition = True

...         target_seq = np.zeros((1, 1, output_vocab_size))  # <6>
...         target_seq[0, 0, generated_token_idx] = 1.
...         thought = [h, c]  # <7>

...     return generated_sequence
"""
