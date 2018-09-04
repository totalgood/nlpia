from nlpia.loaders import get_data


df = get_data('deu')
input_texts, target_texts = [], []  # <1>
input_vocabulary = set()  # <3>
output_vocabulary = set()
start_token = '\t'  # <4>
stop_token = '\n'
max_training_samples = min(25000, len(df) - 1)  # <6>

for input_text, target_text in zip(df.eng, df.deu):
    target_text = start_token + target_text \
        + stop_token  # <7>
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:  # <8>
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)

input_vocabulary = sorted(input_vocabulary)  # <1>
output_vocabulary = sorted(output_vocabulary)

input_vocab_size = len(input_vocabulary)  # <2>
output_vocab_size = len(output_vocabulary)
max_encoder_seq_length = max(
    [len(txt) for txt in input_texts])  # <3>
max_decoder_seq_length = max(
    [len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in
                          enumerate(input_vocabulary)])  # <4>
target_token_index = dict(
    [(char, i) for i, char in enumerate(output_vocabulary)])
reverse_input_char_index = dict((i, char) for char, i in
                                input_token_index.items())  # <5>
reverse_target_char_index = dict((i, char) for char, i in
                                 target_token_index.items())

import numpy as np  # <1>  # noqa


encoder_input_data = np.zeros((len(input_texts),
                               max_encoder_seq_length, input_vocab_size),
                              dtype='float32')  # <2>
decoder_input_data = np.zeros((len(input_texts),
                               max_decoder_seq_length, output_vocab_size),
                              dtype='float32')
decoder_target_data = np.zeros((len(input_texts),
                                max_decoder_seq_length, output_vocab_size),
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
