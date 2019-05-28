# import os
# import re
import sys

# import pandas as pd
import numpy as np
# from tqdm import tqdm
from nlpia.loaders import get_data

if len(sys.argv) > 1:
    lang = sys.argv[1][:3].lower()
else:
    lang = 'spa'

source_lang = 'eng'
max_text_len = 32  # 32: 62,829 spa examples (2 min/epoch), 64: 100k spa examples (6 min/epoch)


df = get_data(lang)
if lang not in df.columns:
    # print(df.columns)
    print(f"changing language name {lang} to {list(df.columns)[-1]}")
    lang = list(df.columns)[-1]


df.columns = [source_lang, lang]
df = df.dropna()
df[source_lang + '_len'] = df[source_lang].str.len()
df = df[df[source_lang + '_len'] < max_text_len]
# need to make sure all batches are composed of similar-length texts (see NLPIA ch 10 & ch 13)

input_texts, target_texts = [], []  # <1>
start_token, stop_token = '\t\n'  # <3>
input_vocab = set()  # <2>
output_vocab = set(start_token + stop_token)
n_samples = min(100000, len(df))  # <4>

df['target'] = start_token + df[lang] + stop_token
for statement in df[source_lang]:
    input_vocab.update(set(statement))
for reply in df[lang]:
    output_vocab.update(set(reply))
input_vocab = tuple(sorted(input_vocab))
output_vocab = tuple(sorted(output_vocab))

max_encoder_seq_len = df[source_lang].str.len().max()
# max_encoder_seq_len
# 100
max_decoder_seq_len = df.target.str.len().max()
# max_decoder_seq_len
# 102

""" Construct character sequence encoder-decoder training set
"""
import numpy as np  # <1> # noqa

encoder_input_onehot = np.zeros(
    (len(df), max_encoder_seq_len, len(input_vocab)),
    dtype='float32')  # <2>
decoder_input_onehot = np.zeros(
    (len(df), max_decoder_seq_len, len(output_vocab)),
    dtype='float32')
decoder_target_onehot = np.zeros(
    (len(df), max_decoder_seq_len, len(output_vocab)),
    dtype='float32')

for i, (input_text, target_text) in enumerate(
        zip(df[source_lang], df.target)):  # <3>
    for t, c in enumerate(input_text):  # <4>
        k = input_vocab.index(c)
        encoder_input_onehot[i, t, k] = 1.  # <5>
    k = np.array([output_vocab.index(c) for c in target_text])
    decoder_input_onehot[i, np.arange(len(target_text)), k] = 1.
    decoder_target_onehot[i, np.arange(len(target_text) - 1), k[1:]] = 1.
# <1> You use numpy for the matrix manipulations.
# <2> The training tensors are initialized as zero tensors with the shape of number of samples
#     (this number should be equal for the input and target samples) times the maximum number of sequence tokens
#     times the number of possible characters.
# <3> Loop over the training samples; input and target texts need to match.
# <4> Loop over each character of each sample.
# <5> Set the index for the character at each time step to one; all other indices remain at zero.
#     This creates the one-hot encoded representation of the training samples.
# <6> For the training data for the decoder, you create the `decoder_input_data` and `decoder_target_data`
#     (which is one time step behind the _decoder_input_data_).


"""Construct and train a character sequence encoder-decoder network
"""
from keras.models import Model  # noqa
from keras.layers import Input, LSTM, Dense  # noqa

batch_size = 64    # <1>
epochs = 15       # <2>
num_neurons = 256  # <3>

encoder_inputs = Input(shape=(None, len(input_vocab)))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, len(output_vocab)))
decoder_lstm = LSTM(num_neurons, return_sequences=True,
                    return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(output_vocab), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['acc'])


model.fit([encoder_input_onehot, decoder_input_onehot],
          decoder_target_onehot, batch_size=batch_size, epochs=epochs,
          validation_split=0.10)  # <4>
# 57915/57915 [==============================] - 296s 5ms/step - loss: 0.7575 - acc: 0.1210 - val_loss: 0.6521 - val_acc: 0.1517
# Epoch 2/100
# 57915/57915 [==============================] - 283s 5ms/step - loss: 0.5924 - acc: 0.1613 - val_loss: 0.5738 - val_acc: 0.1734
# ...
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4235 - acc: 0.2075 - val_loss: 0.4688 - val_acc: 0.2034
# Epoch 21/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4217 - acc: 0.2080 - val_loss: 0.4680 - val_acc: 0.2037
# Epoch 22/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4198 - acc: 0.2084 - val_loss: 0.4686 - val_acc: 0.2035
# ...
# Epoch 69/100                                                                                                                                                                                                                           [1480/1902]
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3830 - acc: 0.2191 - val_loss: 0.4912 - val_acc: 0.2008
# Epoch 70/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3826 - acc: 0.2193 - val_loss: 0.4902 - val_acc: 0.2007
# Epoch 71/100
# ...
# Epoch 99/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3738 - acc: 0.2220 - val_loss: 0.5000 - val_acc: 0.1994
# Epoch 100/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3736 - acc: 0.2220 - val_loss: 0.5017 - val_acc: 0.1992

""" .Construct response generator model
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

...     target_seq = np.zeros((1, 1, len(output_vocab)))  # <2>
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
...                 len(generated_sequence) > max_decoder_seq_len
...                 ):  # <5>
...             stop_condition = True

...         target_seq = np.zeros((1, 1, len(output_vocab)))  # <6>
...         target_seq[0, 0, generated_token_idx] = 1.
...         thought = [h, c]  # <7>

...     return generated_sequence
"""
def decode_sequence(input_seq):
    thought = encoder_model.predict(input_seq)  # <1>

    target_seq = np.zeros((1, 1, len(output_vocab)))  # <2>
    target_seq[0, 0, output_vocab.index(stop_token)] = 1.  # <3>
    stop_condition = False
    generated_sequence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + thought) # <4>

        # take only the most likely character (temperature = 0)
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = output_vocab[generated_token_idx]
        generated_sequence += generated_char
        if (generated_char == stop_token or
                len(generated_sequence) > max_decoder_seq_len
                ):  # <5>
            stop_condition = True

        target_seq = np.zeros((1, 1, len(output_vocab)))  # <6>
        target_seq[0, 0, generated_token_idx] = 1.
        thought = [h, c]  # <7>

    return generated_sequence


def respond(input_text, input_vocab=input_vocab):
    input_text = input_text.lower()
    input_text = ''.join(c if c in input_vocab else ' ' for c in input_text)
    input_seq = np.zeros((1, max_encoder_seq_len, len(input_vocab)), dtype='float32')
    for t, c in enumerate(input_text):
        input_seq[0, t, input_vocab.index(c)] = 1.
    decoded_sentence = decode_sequence(input_seq)
    print('Human: {}'.format(input_text))
    print('Bot:', decoded_sentence)
    return decoded_sentence
