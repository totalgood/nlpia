#!/usr/bin/env python

"""
.Load word2vec vectors
[source,python]
----
>>> from nlpia.loaders import get_data
>>> wv = get_data('word2vec')  # <1>
100%|############################| 402111/402111 [01:02<00:00, 6455.57it/s]
>>> len(wv.vocab), len(wv[next(iter(wv.vocab))])
(3000000, 300)
>>> wv.vectors.shape
(3000000, 300)
----
<1> If you haven't already downloaded GoogleNews-vectors-negative300.bin.gz (bit.ly/GoogleNews-vectors-negative300) to `nlpia/src/nlpia/bigdata/` then `get_data()` will download it for you.
"""
from nlpia.loaders import get_data
wv = get_data('word2vec')  # <1>
# 100%|############################| 402111/402111 [01:02<00:00, 6455.57it/s]
len(wv.vocab), len(wv[next(iter(wv.vocab))])
# (3000000, 300)
wv.vectors.shape
# (3000000, 300)


"""
.Add each word vector to the `AnnoyIndex`
[source,python]
----
>>> from tqdm import tqdm  # <1>
>>> for i, word in enumerate(tqdm(wv.index2word)):  # <2>
...     index.add_item(i, wv[word])
22%|#######▏                   | 649297/3000000 [00:26<01:35, 24587.52it/s]
----
<1> `tqdm()` takes an iterable and returns an iterable (like `enumerate()`) and inserts code in your loop to display a progress bar
<2> `.index2word` is an unsorted list of all 3M tokens in your vocabulary, equivalent to a map of the integer indexes (0-2999999) to tokens ('</s>' to 'snowcapped_Caucasus').
"""
from annoy import AnnoyIndex
num_words, num_dimensions = wv.vectors.shape  # <1>
index = AnnoyIndex(num_dimensions)

"""
>>> from tqdm import tqdm
>>> for i, word in enumerate(tqdm(wv.index2word)):
...     index.add_item(i, wv[word])
"""
from tqdm import tqdm
for i, word in enumerate(tqdm(wv.index2word)):
    index.add_item(i, wv[word])


"""
.Build Euclidean distance index with 15 trees

>>> import numpy as np
>>> num_vectors = len(wv.vocab)
>>> num_trees = int(np.log(num_vectors).round(0))  # <1>
>>> num_trees
15
>>> index.build(num_trees)  # <2>
>>> index.save('Word2vec_index.ann')  # <3>
True
>>> w2id = dict(zip(range(len(wv.vocab)), wv.vocab))

<1> This is just a rule of thumb -- you may want to optimize this hyperparameter if this index isn't performant for the things you care about (RAM, lookup, indexing) or accurate enough for your application.
<2> round(ln(3000000)) => 15 indexing trees for our 3M vectors -- this takes a few minutes on a laptop
<3> Saves the index to a local file and frees up RAM
"""
import numpy as np
num_trees = int(np.log(num_words).round(0))  # <1>
index.build(num_trees)  # <2>
index.save('Word2vec_index.ann')  # <3>


"""
>>> wv.vocab['Harry_Potter'].index  # <1>
9494
>>> wv.vocab['Harry_Potter'].count  # <2>
2990506
>>> w2id = dict(zip(
...     wv.vocab, range(len(wv.vocab))))  # <3>
>>> w2id['Harry_Potter']
9494
>>> ids = index.get_nns_by_item(
...     w2id['Harry_Potter'], 11)  # <4>
>>> ids
[9494, 32643, 39034, 114813, ..., 113008, 116741, 113955, 350346]
>>> [wv.vocab[i] for i in ids]
>>> [wv.index2word[i] for i in ids]
['Harry_Potter',
 'Narnia',
 'Sherlock_Holmes',
 'Lemony_Snicket',
 'Spiderwick_Chronicles',
 'Unfortunate_Events',
 'Prince_Caspian',
 'Eragon',
 'Sorcerer_Apprentice',
 'RL_Stine']
"""
wv.vocab['Harry_Potter'].index  # <1>
# 9494
wv.vocab['Harry_Potter'].count  # <2>
# 2990506
w2id = dict(zip(
    wv.vocab, range(len(wv.vocab))))  # <3>
w2id['Harry_Potter']
# 9494
ids = index.get_nns_by_item(
    w2id['Harry_Potter'], 11)  # <4>
ids
[9494, 32643, 39034, 114813, ..., 113008, 116741, 113955, 350346]
[wv.vocab[i] for i in ids]
[wv.index2word[i] for i in ids]
['Harry_Potter',
 'Narnia',
 'Sherlock_Holmes',
 'Lemony_Snicket',
 'Spiderwick_Chronicles',
 'Unfortunate_Events',
 'Prince_Caspian',
 'Eragon',
 'Sorcerer_Apprentice',
 'RL_Stine']
