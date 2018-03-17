from __future__ import absolute_import, division, print_function, unicode_literals
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import *  # noqa
import tempfile

import pandas as pd
from nlpia.constants import logging

from annoy import AnnoyIndex

np = pd.np
logger = logging.getLogger(__name__)


def representative_sample(X, num_samples, save=False):
    """Sample vectors in X, prefering edge cases and vectors farthest from other vectors in sample set


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
        if isinstance(save, (bytes, str)):
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
        # FIXME: pick vector furthest from past K (K > 1) points or outside of a hypercube (sized to uniformly fill the space) around the last sample
        try:
            samples[j + 1] = np.setdiff1d(nns, samples)[-1]
        except:
            samples[j + 1]
        if len(num_nns) < num_samples / 3.:
            num_nns = min(N, 1.3 * num_nns)
        j += 1
