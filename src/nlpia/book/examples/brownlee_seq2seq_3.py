# from random import randint
# from numpy import array
# from numpy import argmax
# from numpy import array_equal
# from keras.models import Sequential
# from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import Input
from keras.layser import Embedding  # Dropout, Activation, Embedding
from keras.models import Model  # Sequential
from keras.layers.merge import Concatenate


# Alternative 3
def build_model(vocab_size=None, src_txt_length=None, sum_txt_length=None):
    # document input model
    concatenate = Concatenate()
    inputs1 = Input(shape=(src_txt_length,))
    article1 = Embedding(vocab_size, 128)(inputs1)
    article2 = LSTM(128)(article1)
    article3 = RepeatVector(sum_txt_length)(article2)
    # summary input model
    inputs2 = Input(shape=(sum_txt_length,))
    summ1 = Embedding(vocab_size, 128)(inputs2)
    # decoder model
    decoder1 = concatenate([article3, summ1])
    decoder2 = LSTM(128)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [article, summary] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
