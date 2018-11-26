""" Build character sequence-to-sequence training set
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
from nlpia.loaders import get_data
df = get_data(os.path.join('..', 'book', 'data', 'dialog.txt'))
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


""" Character sequence-to-sequence model parameters
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


""" Construct character sequence encoder-decoder training set

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


"""
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
