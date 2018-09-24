import os

import numpy as np
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nlpia.constants import BIGDATA_PATH
from nlpia.loaders import get_data
from pugnlp.futil import mkdir_p

MAX_NUM_WORDS = 1000000
EMBEDDING_DIM = 300
MAX_INPUT_SEQUENCE_LENGTH = 256
MAX_TARGET_SEQUENCE_LENGTH = 256


def onehot_char_training_data(lang='deu', n=700, data_paths=()):
    df = get_data(lang)
    n = int(len(df) * n) if n <= 1 else n 
    df = df.iloc[:n]
    input_texts, target_texts = [], []  # <1>
    input_vocabulary = set()  # <3>
    output_vocabulary = set()
    start_token, stop_token = '\t\n'  # <2>
    n = len(df)

    for input_text, target_text in tqdm(zip(df.eng, df[lang]), total=n):
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

    trainset = (encoder_input_data, decoder_input_data, decoder_target_data)
    for i, p in enumerate(data_paths):
        np.save(p, trainset[i][:n], allow_pickle=False)

    return encoder_input_data, decoder_input_data, decoder_target_data


def wordvector_training_data(lang='deu', n=700, data_paths=()):
    df = get_data(lang)
    n = int(len(df) * n) if n <= 1 else n 
    n = min(len(df), n)
    df = df.iloc[:n]
    input_texts, target_texts = [], []  # <1>
    input_vocabulary = set()  # <3>
    output_vocabulary = set()
    start_token, stop_token = '<START>', '<STOP>'
    input_tokenizer, output_tokenizer = Tokenizer(), Tokenizer()
    wv = get_data('word2vec')
    EMBEDDING_DIM = len(wv['the'])

    for input_text, target_text in tqdm(zip(df.eng, df[lang]), total=n):
        target_text = start_token + target_text + stop_token
        input_texts.append(input_text)
        target_texts.append(target_text)

    # texts = input_texts + target_texts
    # assert(len(texts) == n * 2)
    # input_texts = texts[:n]
    # target_texts = texts[n:]

    input_tokenizer.fit_on_texts(input_texts)
    output_tokenizer.fit_on_texts(target_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    target_sequences = output_tokenizer.texts_to_sequences(target_texts)
    input_sequences = pad_sequences(input_sequences, maxlen=MAX_INPUT_SEQUENCE_LENGTH)
    target_sequences = pad_sequences(target_sequences, maxlen=MAX_TARGET_SEQUENCE_LENGTH)

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    for w, i in input_tokenizer.word_index.items():
        if w in wv.vocab:
            embedding_matrix[i] = wv.word_vec(w)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix != 0, axis=1) == 0))

#     input_vocabulary = sorted(input_vocabulary)  # <1>
#     output_vocabulary = sorted(output_vocabulary)

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

    trainset = (encoder_input_data, decoder_input_data, decoder_target_data)
    for i, p in enumerate(data_paths):
        np.save(p, trainset[i][:n], allow_pickle=False)

    return encoder_input_data, decoder_input_data, decoder_target_data


def fit(
        data_paths=(),
        epochs=100,
        batch_size=64,
        num_neurons=256,
        ):

    encoder_input_data = np.load(data_paths[0])
    decoder_input_data = np.load(data_paths[1])
    decoder_target_data = np.load(data_paths[2])

    input_vocab_size = encoder_input_data.shape[2]
    output_vocab_size = decoder_target_data.shape[2]

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

    checkpoint_path = os.path.join(BIGDATA_PATH, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, 'nlpia-seq2seq-translation-weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              callbacks=[checkpoint_callback],
              batch_size=batch_size, epochs=epochs, validation_split=0.1)  # <4>

    return model


def main(
        lang='deu', n=900, epochs=50, batch_size=64, num_neurons=256,
        encoder_input_data=None,
        decoder_input_data=None,
        decoder_target_data=None,
        checkpoint_dir=os.path.join(BIGDATA_PATH, 'checkpoints'),
        ):
    """ Train an LSTM encoder-decoder squence-to-sequence model on Anki flashcards for international translation

    >>> model = main('spa', n=400, epochs=3, batch_size=128, num_neurons=32)
    Train on 360 samples, validate on 40 samples
    Epoch 1/3
    ...
    >>> len(model.get_weights())
    8

    # 64 common characters in German, 56 in English
    >>> model.get_weights()[-1].shape[0] >=50
    True
    >>> model.get_weights()[-2].shape[0]
    32
    """
    mkdir_p(checkpoint_dir)
    encoder_input_path = os.path.join(
        checkpoint_dir,
        'nlpia-ch10-translate-input-{}.npy'.format(lang))
    decoder_input_path = os.path.join(
        checkpoint_dir,
        'nlpia-ch10-translate-decoder-input-{}.npy'.format(lang))
    decoder_target_path = os.path.join(
        checkpoint_dir,
        'nlpia-ch10-translate-target-{}.npy'.format('eng'))
    data_paths = (encoder_input_path, decoder_input_path, decoder_target_path)

    encoder_input_data = []
    if all([os.path.isfile(p) for p in data_paths]):
        encoder_input_data = np.load(encoder_input_path)
        decoder_input_data = np.load(decoder_input_path)
        decoder_target_data = np.load(decoder_target_path)
    if len(encoder_input_data) < n:
        encoder_input_data, decoder_input_data, decoder_target_data = onehot_char_training_data(
            lang=lang, n=n, data_paths=data_paths)
    encoder_input_data = encoder_input_data[:n]
    decoder_input_data = decoder_input_data[:n] 
    decoder_target_data = decoder_target_data[:n]
    model = fit(data_paths=data_paths, epochs=epochs, batch_size=batch_size, num_neurons=num_neurons)
    return model


if __name__ == '__main__':
    main()
