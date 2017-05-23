#!usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str,  object  # noqa
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

import pandas as pd

import future        # noqa
import builtins      # noqa
import past          # noqa
import six           # noqa
from lshash.lshash import LSHash

np = pd.np


def sphere():
    X = np.random.normal(size=(1000, 3))
    lsh = LSHash(10, 3, num_hashtables=5)
    for x in X:
        x /= np.linalg.norm(x)
        lsh.index(x)
    closest = lsh.query(X[0] + np.array([-0.001, 0.001, -0.001]), distance_func="cosine")
    assert len(closest) >= 10
    assert 0.05 >= closest[9][-1] > 0.0003


def hyperspheres(X=16, num_samples=200000):
    """ Demonstrate curse of dimensionality and where LSH starts to fail

    Returns:
      lsh, X, secondclosest, tenthclosest

    >>> import pandas as pd
    >>> lsh, vectors, dfs = hyperspheres(16)
    >>> for df in dfs:
    ...     print(df)
    """
    X = np.random.uniform(size=(num_samples, X)) if isinstance(X, int) else X
    closest = []
    secondclosest = []
    tenthclosest = []
    hundredthclosest = []
    for D in range(2, X.shape[1] + 1):
        lsh = LSHash(int(64 / D) + D, D, num_hashtables=D)

        # query vector
        q = np.random.uniform(size=(D,))
        q /= np.linalg.norm(q)

        distances = []
        for x in X[:, :D]:
            x /= np.linalg.norm(x)
            distances += [1. - np.sum(x * q)]  # cosine similarity
            lsh.index(x)
        distances = sorted(distances)
        print(distances[:10])
        closest10 = lsh.query(q, distance_func='cosine')

        N = len(closest10)
        hundredthclosest += [[D, N, closest10[min(99, N - 1)][-1] if N else 2., distances[min(99, N - 1)]]]
        tenthclosest += [[D, N, closest10[min(9, N - 1)][-1] if N else 2., distances[min(9, N - 1)]]]
        secondclosest += [[D, N, closest10[min(1, N - 1)][-1] if N else 2., distances[min(1, N - 1)]]]
        closest += [[D, N, closest10[0][-1] if N else 2., distances[0]]]
        print("is correct: 100th 10th 2nd 1st")
        print(round(hundredthclosest[-1][-1], 14) == round(hundredthclosest[-1][-2], 14))
        print(round(tenthclosest[-1][-1], 14) == round(tenthclosest[-1][-2], 14))
        print(round(secondclosest[-1][-1], 14) == round(secondclosest[-1][-2], 14))
        print(round(closest[-1][-1], 14) == round(closest[-1][-2], 14))
        print("distances: 100th 10th 2nd 1st")
        print(hundredthclosest[-1])
        print(tenthclosest[-1])
        print(secondclosest[-1])
        print(closest[-1])
    dfs = []
    for k, (i, df) in enumerate(zip([100, 10, 2, 1], [hundredthclosest, tenthclosest, secondclosest, closest])):
        df = pd.DataFrame(df, columns='D N dist{} true_dist{}'.format(i, i).split()).round(14)
        df['correct{}'.format(i)] = df['dist{}'.format(i)] == df['true_dist{}'.format(i)]
        dfs += [df]
    # for i, tc in enumerate(tenthclosest):
    #     assert 1e-9 < tc[-2] or 1e-6 < 0.2
    return lsh, X, dfs
