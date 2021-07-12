""" NLPIA chapter 6 glove nessvectors

Dependencies:
  * python==3.6.12
  * scann==

References:
  * Stanford NLP's GloVe model and training script [https://github.com/stanfordnlp/glove]
  * Erik Bern's ANN benchmarks with training and testsets: https://github.com/erikbern/ann-benchmarks
  * Spotify's Annoy (with good visualization): [https://github.com/spotify/annoy]
  * Google Research's ScaNN: [pip install scann]()
"""


import np


def load_glove(filepath):
    # print("Loading Glove Model")
    f = open(filepath, 'r')
    wv = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        embedding = np.array([float(value) for value in splitLines[1:]])
        wv[word] = embedding
    # print(len(wv), " words loaded!")
    return wv
