r""" Build character sequence-to-sequence training set

>>> df = get_data('moviedialog')
>>> df.columns = 'statement reply'.split()
>>> df = df.dropna()
>>> input_texts, target_texts = [], []  # <1>
>>> start_token, stop_token = '\t', '\n'  # <3>
>>> input_vocab = set(start_token+stop_token)  # <2>
>>> output_vocab = set()
>>> n_samples = min(100000, len(df))  # <4>

>>> df['target'] = start_token + df.reply + stop_token 
>>> for statement in df.statement:
...     input_vocab.update(set(statement))
>>> for reply in df.reply:
...     output_vocab.update(set(reply))
>>> input_vocab = tuple(sorted(input_vocab))
>>> output_vocab = tuple(sorted(output_vocab))
>>> input_vocabulary = tuple(sorted(input_vocab)) 
>>> output_vocabulary = tuple(sorted(output_vocab))

>>> max_encoder_seq_len = df.statement.str.len().max()  # <3>
>>> max_decoder_seq_len = df.target.str.len().max()
>>> max_encoder_seq_len, max_decoder_seq_len
(100, 102)
"""
import os
from nlpia.loaders import get_data

df = get_data('moviedialog')
df.columns = 'statement reply'.split()
df = df.dropna()
input_texts, target_texts = [], []  # <1>
start_token, stop_token = '\t\n'  # <3>
input_vocab = set()  # <2>
output_vocab = set(start_token + stop_token)
n_samples = min(100000, len(df))  # <4>

df['target'] = start_token + df.reply + stop_token 
[input_vocab.update(set(statement)) for statement in df.statement]
[output_vocab.update(set(reply)) for reply in df.reply]
input_vocab = tuple(sorted(input_vocab)) #<6>
output_vocab = tuple(sorted(output_vocab))

max_encoder_seq_len = df.statement.str.len().max()
# max_encoder_seq_len
# 100
max_decoder_seq_len = df.target.str.len().max()
# max_decoder_seq_len
# 102

# <1> The arrays hold the input and target text read from the corpus file.
# <2> The sets hold the seen characters in the input and target text.
# <3> The target sequence is annotated with a start (first) and stop (last) token; the characters representing the tokens are defined here. These tokens can't be part of the normal sequence text and should be uniquely used as start and stop tokens.
# <4> `max_training_samples` defines how many lines are used for the training. It is the lower number of either a user-defined maximum or the total number of lines loaded from the file.
# <6> Compile the vocabulary -- set of the unique characters seen in the input_texts


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
        zip(df.statement, df.target)):  # <3>
    for t, c in enumerate(input_text):  # <4>
        encoder_input_onehot[
            i, t, input_vocab.index(c)] = 1.  # <5>
    for t, c in enumerate(target_text):  # <6>
        decoder_input_onehot[
            i, t, output_vocab.index(c)] = 1.
        if t > 0:
            decoder_target_onehot[i, t - 1, output_vocab.index(c)] = 1
# <1> You use numpy for the matrix manipulations.
# <2> The training tensors are initialized as zero tensors with the shape of number of samples (this number should be equal for the input and target samples) times the maximum number of sequence tokens times the number of possible characters.
# <3> Loop over the training samples; input and target texts need to match.
# <4> Loop over each character of each sample.
# <5> Set the index for the character at each time step to one; all other indices remain at zero. This creates the one-hot encoded representation of the training samples.
# <6> For the training data for the decoder, you create the `decoder_input_data` and `decoder_target_data` (which is one time step behind the _decoder_input_data_).


"""Construct and train a character sequence encoder-decoder network
"""
from keras.models import Model  # noqa
from keras.layers import Input, LSTM, Dense  # noqa

batch_size = 64    # <1>
epochs = 100       # <2>
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
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4826 - acc: 0.1911 - val_loss: 0.4972 - val_acc: 0.1953
# Epoch 7/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4728 - acc: 0.1938 - val_loss: 0.4908 - val_acc: 0.1970
# Epoch 8/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4661 - acc: 0.1956 - val_loss: 0.4879 - val_acc: 0.1976
# Epoch 9/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4594 - acc: 0.1974 - val_loss: 0.4826 - val_acc: 0.1990
# Epoch 10/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4537 - acc: 0.1990 - val_loss: 0.4798 - val_acc: 0.2001
# Epoch 11/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4490 - acc: 0.2002 - val_loss: 0.4768 - val_acc: 0.2006
# Epoch 12/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4450 - acc: 0.2014 - val_loss: 0.4751 - val_acc: 0.2010
# Epoch 13/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4415 - acc: 0.2022 - val_loss: 0.4744 - val_acc: 0.2015
# Epoch 14/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4382 - acc: 0.2033 - val_loss: 0.4723 - val_acc: 0.2020
# Epoch 15/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4352 - acc: 0.2041 - val_loss: 0.4706 - val_acc: 0.2027
# Epoch 16/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4324 - acc: 0.2049 - val_loss: 0.4704 - val_acc: 0.2027
# Epoch 17/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4299 - acc: 0.2056 - val_loss: 0.4700 - val_acc: 0.2030
# Epoch 18/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4276 - acc: 0.2062 - val_loss: 0.4689 - val_acc: 0.2035
# Epoch 19/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4255 - acc: 0.2069 - val_loss: 0.4691 - val_acc: 0.2037
# Epoch 20/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4235 - acc: 0.2075 - val_loss: 0.4688 - val_acc: 0.2034
# Epoch 21/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4217 - acc: 0.2080 - val_loss: 0.4680 - val_acc: 0.2037
# Epoch 22/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4198 - acc: 0.2084 - val_loss: 0.4686 - val_acc: 0.2035
# Epoch 23/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4182 - acc: 0.2089 - val_loss: 0.4697 - val_acc: 0.2032
# Epoch 24/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4166 - acc: 0.2094 - val_loss: 0.4699 - val_acc: 0.2032
# Epoch 25/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4151 - acc: 0.2098 - val_loss: 0.4695 - val_acc: 0.2035
# Epoch 26/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4137 - acc: 0.2102 - val_loss: 0.4697 - val_acc: 0.2037
# Epoch 27/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4123 - acc: 0.2106 - val_loss: 0.4700 - val_acc: 0.2038
# Epoch 28/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4110 - acc: 0.2110 - val_loss: 0.4708 - val_acc: 0.2036
# Epoch 29/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4098 - acc: 0.2113 - val_loss: 0.4706 - val_acc: 0.2036
# Epoch 30/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4086 - acc: 0.2117 - val_loss: 0.4719 - val_acc: 0.2030
# Epoch 31/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.4075 - acc: 0.2120 - val_loss: 0.4717 - val_acc: 0.2032
# Epoch 32/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.4064 - acc: 0.2123 - val_loss: 0.4717 - val_acc: 0.2035
# Epoch 33/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.4053 - acc: 0.2127 - val_loss: 0.4729 - val_acc: 0.2032
...
# Epoch 69/100                                                                                                                                                                                                                           [1480/1902]
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3830 - acc: 0.2191 - val_loss: 0.4912 - val_acc: 0.2008
# Epoch 70/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3826 - acc: 0.2193 - val_loss: 0.4902 - val_acc: 0.2007
# Epoch 71/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3821 - acc: 0.2195 - val_loss: 0.4909 - val_acc: 0.2007
# Epoch 72/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3819 - acc: 0.2196 - val_loss: 0.4921 - val_acc: 0.2009
# Epoch 73/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3815 - acc: 0.2197 - val_loss: 0.4930 - val_acc: 0.2002
# Epoch 74/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3812 - acc: 0.2198 - val_loss: 0.4940 - val_acc: 0.1998
# Epoch 75/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3808 - acc: 0.2199 - val_loss: 0.4940 - val_acc: 0.2000
# Epoch 76/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3804 - acc: 0.2200 - val_loss: 0.4939 - val_acc: 0.2003
# Epoch 77/100
# 57915/57915 [==============================] - 276s 5ms/step - loss: 0.3800 - acc: 0.2201 - val_loss: 0.4951 - val_acc: 0.2001
# Epoch 78/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3798 - acc: 0.2201 - val_loss: 0.4943 - val_acc: 0.2003
# Epoch 79/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3795 - acc: 0.2202 - val_loss: 0.4955 - val_acc: 0.1998
# Epoch 80/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3791 - acc: 0.2203 - val_loss: 0.4951 - val_acc: 0.2001
# Epoch 81/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3787 - acc: 0.2204 - val_loss: 0.4955 - val_acc: 0.1998
# Epoch 82/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3784 - acc: 0.2205 - val_loss: 0.4965 - val_acc: 0.1998
# Epoch 83/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3781 - acc: 0.2207 - val_loss: 0.4965 - val_acc: 0.1996
# Epoch 84/100
# 57915/57915 [==============================] - 279s 5ms/step - loss: 0.3777 - acc: 0.2208 - val_loss: 0.4971 - val_acc: 0.1997
# Epoch 85/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3775 - acc: 0.2209 - val_loss: 0.4969 - val_acc: 0.1996
# Epoch 86/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3773 - acc: 0.2209 - val_loss: 0.4971 - val_acc: 0.1999
# Epoch 87/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3770 - acc: 0.2210 - val_loss: 0.4978 - val_acc: 0.1994
# Epoch 88/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3766 - acc: 0.2211 - val_loss: 0.4983 - val_acc: 0.1996
# Epoch 89/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3764 - acc: 0.2212 - val_loss: 0.4983 - val_acc: 0.1993
# Epoch 90/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3761 - acc: 0.2212 - val_loss: 0.4987 - val_acc: 0.1993
# Epoch 91/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3759 - acc: 0.2213 - val_loss: 0.4989 - val_acc: 0.1993
# Epoch 92/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3756 - acc: 0.2214 - val_loss: 0.4989 - val_acc: 0.1993
# Epoch 93/100
# 57915/57915 [==============================] - 274s 5ms/step - loss: 0.3751 - acc: 0.2215 - val_loss: 0.4996 - val_acc: 0.1991
# Epoch 94/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3751 - acc: 0.2216 - val_loss: 0.4990 - val_acc: 0.1996
# Epoch 95/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3748 - acc: 0.2218 - val_loss: 0.4997 - val_acc: 0.1994
# Epoch 96/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3747 - acc: 0.2216 - val_loss: 0.4999 - val_acc: 0.1990
# Epoch 97/100
# 57915/57915 [==============================] - 277s 5ms/step - loss: 0.3743 - acc: 0.2218 - val_loss: 0.5001 - val_acc: 0.1991
# Epoch 98/100
# 57915/57915 [==============================] - 278s 5ms/step - loss: 0.3742 - acc: 0.2218 - val_loss: 0.5004 - val_acc: 0.1990
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
...                 len(generated_sequence) > max_decoder_seq_length
...                 ):  # <5>
...             stop_condition = True

...         target_seq = np.zeros((1, 1, len(output_vocab)))  # <6>
...         target_seq[0, 0, generated_token_idx] = 1.
...         thought = [h, c]  # <7>

...     return generated_sequence
"""