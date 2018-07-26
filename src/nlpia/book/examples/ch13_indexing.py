import os
from nlpia.data.loaders import BIGDATA_PATH
from gensim.models import KeyedVectors
path = os.path.join(BIGDATA_PATH, 'GoogleNews-vectors-negative300.bin.gz')
wv = KeyedVectors.load_word2vec_format(path, binary=True)
len(wv.vocab)
# 3000000


from annoy import AnnoyIndex
index = AnnoyIndex(f=len(wv[wv.index2word[0]]))
for i, word in enumerate(wv.index2word):
    if not i % 100000:
        print('{}: {}'.format(i, word))
    index.add_item(i, wv[word])
# 0: </s>
# 100000: distinctiveness
# ...
# 2600000: cedar_juniper
# 2700000: Wendy_Liberatore
# 2800000: Management_GDCM
# 2900000: BOARDED_UP


import numpy as np
num_vectors = len(wv.vocab)
num_trees = int(np.log(num_vectors).round(0))
index.build(num_trees)  # <1>
index.save('Word2vec_index.ann')  # <2>
