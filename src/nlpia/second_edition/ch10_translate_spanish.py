import os
import re
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
from nlpia.loaders import get_data

if len(sys.argv) > 1:
    lang = sys.argv[1][:3].lower()
else:
    lang = 'spa'


df = get_data(lang)
if lang not in df.columns:
    # print(df.columns)
    print(f"changing language name {lang} to {list(df.columns)[-1]}")
    lang = list(df.columns)[-1]

input_texts, target_texts = [], []  # <1>
input_vocabulary = set()  # <3>
output_vocabulary = set()
start_token, stop_token = '\t\n'  # <2>
n = int(len(df) * .1)
encoder_input_path = 'encoder_input_data-{}-{}.np'.format(lang, n)
decoder_input_path = 'decoder_input_data-{}-{}.np'.format(lang, n)
decoder_target_path = 'decoder_target_data-eng-{}.np'.format(n)


for k, (input_text, target_text) in enumerate(tqdm(zip(df.eng, df[lang]), total=n)):
    if k == n:
        break
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
input_vocab_filename = (
    f'ch10_translate_eng_input_vocabulary_{len(input_vocabulary)}.txt')
output_vocabulary = sorted(output_vocabulary)
output_vocab_filename = (
    f'ch10_translate_{lang}_output_vocabulary_{len(output_vocabulary)}.txt')

with open(input_vocab_filename, 'w') as fout:
    fout.write(''.join(input_vocabulary))
with open(output_vocab_filename, 'w') as fout:
    fout.write(''.join(output_vocabulary))

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

encoder_input_data = np.zeros((n, max_encoder_seq_length, input_vocab_size),
                              dtype='float32')  # <2>
decoder_input_data = np.zeros((n, max_decoder_seq_length, output_vocab_size),
                              dtype='float32')
decoder_target_data = np.zeros((n, max_decoder_seq_length, output_vocab_size),
                               dtype='float32')
for i, (input_text, target_text) in enumerate(tqdm(
        zip(input_texts, target_texts), total=len(target_texts))):  # <3>
    for t, char in enumerate(input_text):  # <4>
        encoder_input_data[
            i, t, input_token_index[char]] = 1.  # <5>
    for t, char in enumerate(target_text):  # <6>
        decoder_input_data[
            i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1


np.save(encoder_input_path, encoder_input_data, allow_pickle=False)
np.save(decoder_input_path, decoder_input_data, allow_pickle=False)
np.save(decoder_target_path, decoder_target_data, allow_pickle=False)


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

# end constructing model
########################################################################

#########################################################################
# fit the model
from nlpia.constants import BIGDATA_PATH  # noqa
checkpoint_path = os.path.join(BIGDATA_PATH, 'checkpoints')
try:
    os.mkdir(checkpoint_path)
except FileExistsError:
    pass
checkpoint_path = os.path.join(checkpoint_path, 'nlpia_seq2seq_translation_weights-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5')


def fit(
        batch_size=64, epochs=100, validation_split=0.1,
        checkpoint_path=checkpoint_path,
        encoder_input_data=encoder_input_data,
        decoder_input_data=decoder_input_data,
        decoder_target_data=decoder_target_data):
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              callbacks=[checkpoint_callback],
              batch_size=batch_size, epochs=epochs, validation_split=validation_split)  # <4>
    return model
# model = fit(model)
# end fitting the model
#########################################################################


#########################################################################
# use fitted model for prediction/translation...

from keras.callbacks import ModelCheckpoint  # noqa

from keras.models import Model  # noqa
from keras.layers import Input, LSTM, Dense  # noqa


class OneHotEncoder:
    """ Encode text as matrix of one-hot char vectors, and decode back to text

    One column for each letter
    >>> e = OneHotEncoder('abcde')
    >>> onehots = e.encode('cab')
    >>> onehots
    array([[0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.]], dtype=float32)
    >>> e.decode(onehots)
    'cab'
    >>> onehots = e.encode('cast')
    >>> onehots
    array([[0., 0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.]], dtype=float32)
    >>> e.decode(onehots)
    'ca**'
    """

    def __init__(self, vocab):
        if os.path.exists(vocab):
            vocab = open(vocab).read()
        self.vocab = np.array(list(vocab) + ['*'])
        self.index = dict([(c, i) for i, c in enumerate(self.vocab)])

    def encode(self, text):
        onehots = np.zeros((len(text), len(self.vocab)), dtype='float32')
        for i, c in enumerate(text):  # <4>
            onehots[i, self.index.get(c, len(self.vocab) - 1)] = 1.
        return onehots

    def to_onehots(self, text):
        return self.encode(text)

    def decode(self, onehots):
        onehots = onehots  # columns represent characters in the sequence
        text = [' '] * onehots.shape[0]
        for i, mask in enumerate(onehots.astype(bool)):
            text[i] = self.vocab[np.array(mask)][0]
        return ''.join(text)

    def to_text(self, onehots):
        return self.decode(onehots)


input_encoder = OneHotEncoder(vocab=input_vocab_filename)
output_decoder = OneHotEncoder(vocab=output_vocab_filename)


def verify_encoder(
        input_texts=input_texts,
        encoder_input_data=encoder_input_data,
        vocab=input_vocabulary):
    e = OneHotEncoder(vocab=input_vocabulary)
    errors = []
    for text, onehots in zip(input_texts, encoder_input_data):
        errors.append((e.encode(text)[:-1] - onehots).abs().sum())
        if errors[-1]:
            print(e.encode(text))
            print(onehots)
            print()
    print(f'Total Errors: {np.sum(errors)}')
    return errors


class Translator():

    def __init__(self, model, input_vocab=input_vocab_filename, output_vocab=input_vocab_filename):
        if isinstance(model, str) and os.path.exists(model):
            model = Model.load(model)
        self.model = model
        self.input_encoder = OneHotEncoder(vocab=input_vocab)
        self.output_decoder = OneHotEncoder(vocab=output_vocab)

    def interact(self):
        eng = 'x'
        while len(eng.strip()):
            english = input('English: ')
            spanish = self.output_decoder.decode(model.predict(
                self.input_encoder.encode(english)))
            print(f'Spanish: {spanish}')
            print()
        return spanish


def find_best_model(checkpoint_path=os.path.join(BIGDATA_PATH, 'checkpoints')):
    files = os.listdir(checkpoint_path)
    files = pd.DataFrame([[fn] + re.findall(r'[0-9]?[.][0-9]{2}', fn) for fn in files])
    print(files.head())
    files.columns = 'filename epoch loss accuracy'.split()[:len(files.columns)]
    files['goodness'] = np.array(files['accuracy'] if 'accuracy' in files.columns else 1.).astype(float) / files['loss'].astype(float)
    files = files.sort_values('goodness', inplace=False, ascending=False)
    print(files.head())
    checkpoint_path = os.path.join(checkpoint_path, files.iloc[0]['filename'])

    return checkpoint_path


def interract(model, input_encoder, output_decoder):
    eng = ' '
    while len(eng):
        english = input('English: ')
        spanish = output_decoder.decode(model.predict(
            input_encoder.encode(english)))
        print(f'Spanish: {spanish}')
        print()


def compile_model(input_vocab_size=input_vocab_size, output_vocab_size=output_vocab_size):
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
    return model


def fit_and_interact(input_vocab=input_vocab_filename, output_vocab=output_vocab_filename):
    """ load the best model and use it to translate whatever you like """
    input_encoder = OneHotEncoder(vocab=input_vocab_filename)
    output_decoder = OneHotEncoder(vocab=output_vocab_filename)

    # batch_size = 64    # <1>
    # epochs = 100       # <2>
    model = fit()
    print(model)
    model.save(checkpoint_path.format(epoch=999, val_loss=0, val_acc=10))
    model = model.load(find_best_model())
    return interact(model=model, input_encoder=input_encoder, output_decoder=output_decoder)


def load_and_interact(input_vocab=input_vocab_filename, output_vocab=output_vocab_filename):
    """ load the best model and use it to translate whatever you like """
    model = compile_model()
    model.load()
    model = model.load(find_best_model())
    return interact(model=model, input_encoder=input_encoder, output_decoder=output_decoder)


# checkpoint_callback = ModelCheckpoint(checkpoint_path,
#                                       monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
# model.fit([encoder_input_data, decoder_input_data],
#           decoder_target_data,
#           callbacks=[checkpoint_callback],
#           batch_size=batch_size, epochs=epochs, validation_split=0.1)  # <4>

# # load the best model and use it to translate whatever you like
# from keras.models import Model  # noqa
# from keras.layers import Input, LSTM, Dense  # noqa
# from nlpia.constants import BIGDATA_PATH

# batch_size = 64    # <1>
# epochs = 100       # <2>
# num_neurons = 256  # <3>

# encoder_inputs = Input(shape=(None, input_vocab_size))
# encoder = LSTM(num_neurons, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# encoder_states = [state_h, state_c]

# decoder_inputs = Input(shape=(None, output_vocab_size))
# decoder_lstm = LSTM(num_neurons, return_sequences=True,
#                     return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state=encoder_states)
# decoder_dense = Dense(output_vocab_size, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#               metrics=['acc'])

# checkpoint_path = os.path.join(BIGDATA_PATH, 'checkpoints')
# files = os.path.listdir(checkpoint_path)
# files = pd.Dataframe([[fn] + re.findall(fn, r'[.0-9]{2,4}') for fn in files])
# files.columns = 'filename loss accuracy'.split()
# files['goodness'] = files['accuracy'] / files['loss']
# files = files.sort_values('goodness', inplace=False)
# checkpoint_path = os.path.join(checkpoint_path, files.iloc[0]['filename'])

# model = model.load(checkpoint_path)


if __name__ == '__main__':
    fit_and_cli(input_encoder, output_decoder)
