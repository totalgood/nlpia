""" Indexing and searching high dimentional vectors like word embeddings 

Exact KNN solutions are N^2 difficult, but approximate nearest neighbors can be log(N)?

Some High-D vector examples:
- TFIDF vectors (though exact methods are O(log(N)) efficient, due to sparseness)
- LSA topic-document and topic-word vectors
- word embeddings like: Word2vec and GloVE
- sentence encodings like universal sentence encoder output
- thought vectors from LSTM memory cells
- capsule network final layer activations, sentence encodings
- CNN network final layer (before classifcation)
"""
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
from past.builtins import basestring
standard_library.install_aliases()  # noqa

import numpy as np

# difficult to install on Mac OSX
from annoy import AnnoyIndex


def representative_sample(X, num_samples, save=False):
    """Sample vectors in X, preferring edge cases and vectors farthest from other vectors in sample set


    """
    X = X.values if hasattr(X, 'values') else np.array(X)
    N, M = X.shape
    rownums = np.arange(N)
    np.random.shuffle(rownums)

    idx = AnnoyIndex(M)
    for i, row in enumerate(X):
        idx.add_item(i, row)
    idx.build(int(np.log2(N)) + 1)

    if save:
        if isinstance(save, basestring):
            idxfilename = save
        else:
            idxfile = tempfile.NamedTemporaryFile(delete=False)
            idxfile.close()
            idxfilename = idxfile.name
        idx.save(idxfilename)
        idx = AnnoyIndex(M)
        idx.load(idxfile.name)

    samples = -1 * np.ones(shape=(num_samples,), dtype=int)
    samples[0] = rownums[0]
    # FIXME: some integer determined by N and num_samples and distribution
    j, num_nns = 0, min(1000, int(num_samples / 2. + 1))
    for i in rownums:
        if i in samples:
            continue
        nns = idx.get_nns_by_item(i, num_nns)
        # FIXME: pick vector furthest from past K (K > 1) points or outside of a hypercube
        #        (sized to uniformly fill the space) around the last sample
        samples[j + 1] = np.setdiff1d(nns, samples)[-1]
        if len(num_nns) < num_samples / 3.:
            num_nns = min(N, 1.3 * num_nns)
        j += 1
    return samples