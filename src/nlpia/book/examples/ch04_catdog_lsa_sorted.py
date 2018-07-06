""" Uses np.linalg.svd directly to illustrate LSA on a small corpus. 

Examples from the SVD section in Chapter 4 of NLPIA

SEE ALSO: ch04_stanford_lsa.py

>>> from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm
>>> bow_svd, tfidf_svd = lsa_models()  # <1>
>>> prettify_tdm(**bow_svd)
   cat dog apple lion nyc love                                             text
0              1        1                                 NYC is the Big Apple.
1              1        1                        NYC is known as the Big Apple.
2                       1    1                                      I love NYC!
3              1        1           I wore a hat to the Big Apple party in NYC.
4              1        1                       Come to NYC. See the Big Apple!
5              1                             Manhattan is called the Big Apple.
6    1                                  New York is a big city for a small cat.
7    1              1           The lion, a big cat, is the king of the jungle.
8    1                       1                               I love my pet cat.
9                       1    1                      I love New York City (NYC).
10   1   1                                              Your dog chased my cat.
>>> tdm = bow_svd['tdm']
>>> tdm

       0   1   2   3   4   5   6   7   8   9   10
cat     0   0   0   0   0   0   1   1   1   0   1
dog     0   0   0   0   0   0   0   0   0   0   1
apple   1   1   0   1   1   1   0   0   0   0   0
lion    0   0   0   0   0   0   0   1   0   0   0
nyc     1   1   1   1   1   0   0   0   0   1   0
love    0   0   1   0   0   0   0   0   1   1   0
>>> import numpy as np
>>> U, s, Vt = np.linalg.svd(tdm)  # <1>
 
>>> import pandas as pd
>>> pd.DataFrame(U, index=tdm.index).round(2)
          0     1     2     3     4     5
cat   -0.04  0.83 -0.38 -0.00  0.11 -0.38
dog   -0.00  0.21 -0.18 -0.71 -0.39  0.52
apple -0.62 -0.21 -0.51  0.00  0.49  0.27
lion  -0.00  0.21 -0.18  0.71 -0.39  0.52
nyc   -0.75 -0.00  0.24 -0.00 -0.52 -0.32
love  -0.22  0.42  0.69  0.00  0.41  0.37

>>> s.round(1)
array([3.1, 2.2, 1.8, 1. , 0.8, 0.5])
>>> S = np.zeros((len(U), len(Vt)))
>>> pd.np.fill_diagonal(S, s)
>>> pd.DataFrame(S).round(1)
    0    1    2    3    4    5    6    7    8    9    10
0  3.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
1  0.0  2.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
2  0.0  0.0  1.8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
3  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
4  0.0  0.0  0.0  0.0  0.8  0.0  0.0  0.0  0.0  0.0  0.0
5  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0

>>> pd.DataFrame(Vt).round(2)
      0     1     2     3     4     5     6     7     8     9     10
0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01
1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47
2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32
3   0.00 -0.00 -0.00  0.00  0.00  0.00 -0.00  0.71  0.00 -0.00 -0.71
4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33
5  -0.09 -0.09  0.10 -0.09 -0.09  0.51 -0.73  0.27 -0.01  0.10  0.27
6  -0.57  0.21  0.11  0.33 -0.31  0.34  0.34 -0.00 -0.34  0.23  0.00
7  -0.32  0.47  0.25 -0.63  0.41  0.07  0.07  0.00 -0.07 -0.18  0.00
8  -0.50  0.29 -0.20  0.41  0.16 -0.37 -0.37 -0.00  0.37 -0.17  0.00
9  -0.15 -0.15 -0.59 -0.15  0.42  0.04  0.04 -0.00 -0.04  0.63 -0.00
10 -0.26 -0.62  0.33  0.24  0.54  0.09  0.09 -0.00 -0.09 -0.23 -0.00

>>> tdm = bow_svd['tdm']
>>> U, s, Vt = np.linalg.svd(tdm)
>>> S = np.zeros((len(U), len(Vt)))
>>> np.fill_diagonal(S, s)
>>> err = [0]
>>> for numdim in range(len(s), 0, -1):
...     S[numdim - 1, numdim - 1] = 0
...     reconstructed_tdm = U.dot(S).dot(Vt)
...     err.append(np.sqrt(((reconstructed_tdm - tdm).values.flatten() ** 2).sum() 
...                / np.product(tdm.shape)))
>>> np.array(err).round(2)
array([0.  , 0.06, 0.12, 0.17, 0.28, 0.39, 0.55])

>>> tdm = tfidf_svd['tdm']
>>> U, s, Vt = np.linalg.svd(tdm)
>>> S = np.zeros((len(U), len(Vt)))
>>> np.fill_diagonal(S, s)
>>> err2 = [0]
>>> for numdim in range(len(s), 0, -1):
...     S[numdim - 1, numdim - 1] = 0
...     reconstructed_tdm = U.dot(S).dot(Vt)
...     err2.append(np.sqrt(((reconstructed_tdm - tdm).values.flatten() ** 2).sum() 
...                / np.product(tdm.shape)))
>>> np.array(err2).round(2)
array([0.  , 0.07, 0.11, 0.15, 0.23, 0.3 , 0.41])
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # noqa
import seaborn  # noqa
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.data.loaders import get_data

pd.options.display.width = 120
pd.options.display.max_columns = 16

VOCABULARY = vocabulary='cat dog apple lion NYC love'.lower().split()  # 'cat dog apple lion NYC love big small bright'.lower().split()
DOCS = get_data('cats_and_dogs_sorted')


def docs_to_tdm(docs=DOCS, vocabulary=VOCABULARY, verbosity=0):
    tfidfer = TfidfVectorizer(min_df=1, max_df=.99, stop_words=None, token_pattern=r'(?u)\b\w+\b',
                              vocabulary=vocabulary)
    tfidf_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    id_words = [(i, w) for (w, i) in tfidfer.vocabulary_.items()]
    tfidf_dense.columns = list(zip(*sorted(id_words)))[1]

    tfidfer.use_idf = False
    tfidfer.norm = None
    bow_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    bow_dense.columns = list(zip(*sorted(id_words)))[1]
    bow_dense = bow_dense.astype(int)
    tfidfer.use_idf = True
    tfidfer.norm = 'l2'
    if verbosity:
        print(tfidf_dense.T)
    return bow_dense.T, tfidf_dense.T, tfidfer


def prettify_tdm(tdm=None, docs=[], vocabulary=[], **kwargs):
    bow_pretty = tdm.T.copy()[vocabulary]
    bow_pretty['text'] = docs
    for col in vocabulary:
        bow_pretty.loc[bow_pretty[col] == 0, col] = ''
    return bow_pretty


def accuracy_study(tdm=None, u=None, s=None, vt=None, verbosity=0, **kwargs):
    """ Reconstruct the term-document matrix and measure error as SVD terms are truncated
    """
    smat = np.zeros((len(u), len(vt)))
    np.fill_diagonal(smat, s)
    smat = pd.DataFrame(smat, columns=vt.index, index=u.index)
    if verbosity:
        print()
        print('Sigma:')
        print(smat.round(2))
        print()
        print('Sigma without zeroing any dim:')
        print(np.diag(smat.round(2)))
    tdm_prime = u.values.dot(smat.values).dot(vt.values)
    if verbosity:
        print()
        print('Reconstructed Term-Document Matrix')
        print(tdm_prime.round(2))

    err = [np.sqrt(((tdm_prime - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))]
    if verbosity:
        print()
        print('Error without reducing dimensions:')
        print(err[-1])
    # 2.3481474529927113e-15

    smat2 = smat.copy()
    for numdim in range(len(s) - 1, 0, -1):
        smat2.iloc[numdim, numdim] = 0
        if verbosity:
            print('Sigma after zeroing out dim {}'.format(numdim))
            print(np.diag(smat2.round(2)))
            #           d0    d1   d2   d3   d4   d5
            # ship    2.16  0.00  0.0  0.0  0.0  0.0
            # boat    0.00  1.59  0.0  0.0  0.0  0.0
            # ocean   0.00  0.00  0.0  0.0  0.0  0.0
            # voyage  0.00  0.00  0.0  0.0  0.0  0.0
            # trip    0.00  0.00  0.0  0.0  0.0  0.0

        tdm_prime2 = u.values.dot(smat2.values).dot(vt.values)
        err += [np.sqrt(((tdm_prime2 - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))]
        if verbosity:
            print('Error after zeroing out dim {}'.format(numdim))
            print(err[-1])
    return err


def lsa(tdm, verbosity=0):
    if verbosity:
        print(tdm)
        #         0   1   2   3   4   5   6   7   8   9   10
        # cat     0   0   0   0   0   0   1   1   1   0   1
        # dog     0   0   0   0   0   0   0   0   0   0   1
        # apple   1   1   0   1   1   1   0   0   0   0   0
        # lion    0   0   0   0   0   0   0   1   0   0   0
        # love    0   0   1   0   0   0   0   0   1   1   0
        # nyc     1   1   1   1   1   0   0   0   0   1   0

    u, s, vt = np.linalg.svd(tdm)

    u = pd.DataFrame(u, index=tdm.index)
    if verbosity:
        print('U')
        print(u.round(2))
        # U
        #           0     1     2     3     4     5
        # cat   -0.04  0.83 -0.38 -0.00  0.11  0.38
        # dog   -0.00  0.21 -0.18 -0.71 -0.39 -0.52
        # apple -0.62 -0.21 -0.51  0.00  0.49 -0.27
        # lion  -0.00  0.21 -0.18  0.71 -0.39 -0.52
        # love  -0.22  0.42  0.69  0.00  0.41 -0.37
        # nyc   -0.75  0.00  0.24 -0.00 -0.52  0.32

    vt = pd.DataFrame(vt, index=['d{}'.format(i) for i in range(len(vt))])
    if verbosity:
        print('VT')
        print(vt.round(2))
        # VT
        #        0     1     2     3     4     5     6     7     8     9     10
        # d0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01
        # d1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47
        # d2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32
        # d3   0.00 -0.00  0.00  0.00  0.00  0.00 -0.00  0.71  0.00  0.00 -0.71
        # d4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33
        # d5   0.09  0.09 -0.10  0.09  0.09 -0.51  0.73 -0.27  0.01 -0.10 -0.27
        # d6  -0.55  0.24  0.15  0.36 -0.38  0.32  0.32  0.00 -0.32  0.17  0.00
        # d7  -0.32  0.46  0.23 -0.64  0.41  0.09  0.09  0.00 -0.09 -0.14  0.00
        # d8  -0.52  0.27 -0.24  0.39  0.22 -0.36 -0.36 -0.00  0.36 -0.12  0.00
        # d9  -0.14 -0.14 -0.58 -0.14  0.32  0.10  0.10 -0.00 -0.10  0.68 -0.00
        # d10 -0.27 -0.63  0.31  0.23  0.55  0.12  0.12 -0.00 -0.12 -0.19 -0.00

    # Reconstruct the original term-document matrix.
    # The sum of the squares of the error is 0.

    return {'u': u, 's': s, 'vt': vt, 'tdm': tdm}


def plot_feature_selection(accuracy, title=None):
    # accuracy = topic_model['accuracy']
    plt.plot(range(len(accuracy)), accuracy)
    plt.title(title or 'LSA Model Accuracy')
    plt.xlabel('Number of Dimensions Eliminated')
    plt.ylabel('Reconstruction Accuracy')
    plt.grid(True)
    plt.tight_layout()
    return accuracy


""" Some more complicated examples of SVD.
>>> import numpy as np
>>> u, s, vt = np.linalg.svd(tdm)  # <1>

>>> import pandas as pd
>>> u = pd.DataFrame(u, index=tdm.index)
>>> u.round(2)


>>> main('cat dog apple lion NYC love'.lower().split())
263it [00:00, 408405.02it/s]
             0         1         2         3         4    5    6         7         8         9         10   11
nyc    0.674278  0.674278  0.596469  0.674278  0.674278  0.0  0.0  0.000000  0.000000  0.596469  0.000000  0.0
lion   0.000000  0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.826567  0.000000  0.000000  0.000000  0.0
love   0.000000  0.000000  0.802636  0.000000  0.000000  0.0  0.0  0.000000  0.744190  0.802636  0.000000  0.0
dog    0.000000  0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.000000  0.000000  0.000000  0.826567  0.0
apple  0.738477  0.738477  0.000000  0.738477  0.738477  1.0  0.0  0.000000  0.000000  0.000000  0.000000  0.0
cat    0.000000  0.000000  0.000000  0.000000  0.000000  0.0  1.0  0.562839  0.667968  0.000000  0.562839  0.0
   nyc lion love dog apple cat                                             text
0    1                   1                                NYC is the Big Apple.
1    1                   1                       NYC is known as the Big Apple.
2    1         1                                                    I love NYC!
3    1                   1          I wore a hat to the Big Apple party in NYC.
4    1                   1                      Come to NYC. See the Big Apple!
5                        1                   Manhattan is called the Big Apple.
6                            1          New York is a big city for a small cat.
7         1                  1  The lion, a big cat, is the king of the jungle.
8              1             1                               I love my pet cat.
9    1         1                                    I love New York City (NYC).
10                 1         1                          Your dog chased my cat.
11                                                     Bright lights, big city?
       0   1   2   3   4   5   6   7   8   9   10  11
nyc     1   1   1   1   1   0   0   0   0   1   0   0
lion    0   0   0   0   0   0   0   1   0   0   0   0
love    0   0   1   0   0   0   0   0   1   1   0   0
dog     0   0   0   0   0   0   0   0   0   0   1   0
apple   1   1   0   1   1   1   0   0   0   0   0   0
cat     0   0   0   0   0   0   1   1   1   0   1   0
U
          0     1     2     3     4     5
nyc   -0.75 -0.00  0.24  0.00 -0.52 -0.32
lion  -0.00  0.21 -0.18 -0.71 -0.39  0.52
love  -0.22  0.42  0.69 -0.00  0.41  0.37
dog   -0.00  0.21 -0.18  0.71 -0.39  0.52
apple -0.62 -0.21 -0.51 -0.00  0.49  0.27
cat   -0.04  0.83 -0.38 -0.00  0.11 -0.38
VT
       0     1     2     3     4     5     6     7     8     9     10   11
d0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01  0.0
d1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47  0.0
d2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32  0.0
d3   0.00  0.00  0.00  0.00  0.00 -0.00 -0.00 -0.71 -0.00  0.00  0.71  0.0
d4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33  0.0
d5  -0.09 -0.09  0.10 -0.09 -0.09  0.51 -0.73  0.27 -0.01  0.10  0.27  0.0
d6   0.12 -0.03  0.20 -0.03 -0.51  0.45  0.45  0.00 -0.45  0.25  0.00  0.0
d7   0.45 -0.85 -0.04  0.15  0.22  0.02  0.02  0.00 -0.02  0.06  0.00  0.0
d8   0.52  0.15 -0.38  0.15 -0.59 -0.24 -0.24  0.00  0.24  0.14  0.00  0.0
d9  -0.28 -0.01 -0.62 -0.01  0.22  0.07  0.07  0.00 -0.07  0.69  0.00  0.0
d10  0.45  0.15 -0.04 -0.85  0.22  0.02  0.02  0.00 -0.02  0.06  0.00  0.0
d11  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.0
Sigma
         d0    d1    d2   d3    d4    d5   d6   d7   d8   d9  d10  d11
nyc    3.14  0.00  0.00  0.0  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
lion   0.00  2.24  0.00  0.0  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
love   0.00  0.00  1.77  0.0  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
dog    0.00  0.00  0.00  1.0  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
apple  0.00  0.00  0.00  0.0  0.84  0.00  0.0  0.0  0.0  0.0  0.0  0.0
cat    0.00  0.00  0.00  0.0  0.00  0.52  0.0  0.0  0.0  0.0  0.0  0.0
Sigma without zeroing any dim
[3.14 2.24 1.77 1.   0.84 0.52]
Reconstructed Term-Document Matrix
[[ 1.  1.  1.  1.  1.  0. -0. -0. -0.  1. -0.  0.]
 [-0. -0.  0. -0. -0. -0.  0.  1.  0.  0.  0.  0.]
 [ 0.  0.  1.  0.  0.  0. -0. -0.  1.  1. -0.  0.]
 [-0. -0.  0. -0. -0. -0.  0.  0.  0.  0.  1.  0.]
 [ 1.  1.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.]
 [-0. -0.  0. -0. -0. -0.  1.  1.  1.  0.  1.  0.]]
Error without reducing dimensions
1.5846963069762592e-15
Sigma after zeroing out dim 5
[3.14 2.24 1.77 1.   0.84 0.  ]
Error after zeroing out dim 5
0.0614920369471735
Sigma after zeroing out dim 4
[3.14 2.24 1.77 1.   0.   0.  ]
Error after zeroing out dim 4
0.11683863316404541
Sigma after zeroing out dim 3
[3.14 2.24 1.77 0.   0.   0.  ]
Error after zeroing out dim 3
0.1659522675004209
Sigma after zeroing out dim 2
[3.14 2.24 0.   0.   0.   0.  ]
Error after zeroing out dim 2
0.2667342349285279
Sigma after zeroing out dim 1
[3.14 0.   0.   0.   0.   0.  ]
Error after zeroing out dim 1
0.3749554593913143
[1.         0.93850796 0.88316137 0.83404773 0.73326577 0.62504454]
             0         1         2         3         4    5    6         7         8         9         10   11
nyc    0.674278  0.674278  0.596469  0.674278  0.674278  0.0  0.0  0.000000  0.000000  0.596469  0.000000  0.0
lion   0.000000  0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.826567  0.000000  0.000000  0.000000  0.0
love   0.000000  0.000000  0.802636  0.000000  0.000000  0.0  0.0  0.000000  0.744190  0.802636  0.000000  0.0
dog    0.000000  0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.000000  0.000000  0.000000  0.826567  0.0
apple  0.738477  0.738477  0.000000  0.738477  0.738477  1.0  0.0  0.000000  0.000000  0.000000  0.000000  0.0
cat    0.000000  0.000000  0.000000  0.000000  0.000000  0.0  1.0  0.562839  0.667968  0.000000  0.562839  0.0
U
          0     1     2     3     4     5
nyc   -0.66  0.07 -0.27 -0.00  0.51 -0.47
lion  -0.00  0.18  0.21  0.71  0.48  0.44
love  -0.21  0.54 -0.64  0.00 -0.30  0.41
dog   -0.00  0.18  0.21 -0.71  0.48  0.44
apple -0.72 -0.26  0.40 -0.00 -0.38  0.33
cat   -0.04  0.76  0.52 -0.00 -0.19 -0.35
VT
       0     1     2     3     4     5     6     7     8     9     10   11
d0  -0.44 -0.44 -0.25 -0.44 -0.44 -0.32 -0.02 -0.01 -0.08 -0.25 -0.01  0.0
d1  -0.09 -0.09  0.29 -0.09 -0.09 -0.16  0.46  0.35  0.56  0.29  0.35  0.0
d2   0.08  0.08 -0.50  0.08  0.08  0.30  0.38  0.34 -0.09 -0.50  0.34  0.0
d3  -0.00 -0.00  0.00 -0.00 -0.00  0.00 -0.00  0.71  0.00  0.00 -0.71  0.0
d4   0.09  0.09  0.10  0.09  0.09 -0.54 -0.27  0.41 -0.49  0.10  0.41  0.0
d5  -0.13 -0.13  0.08 -0.13 -0.13  0.58 -0.62  0.30  0.13  0.08  0.30  0.0
d6   0.22  0.01  0.20  0.01 -0.64  0.30  0.33  0.00 -0.49  0.26  0.00  0.0
d7   0.46 -0.85 -0.03  0.15  0.21  0.02  0.02  0.00 -0.03  0.06  0.00  0.0
d8   0.49  0.17 -0.42  0.17 -0.49 -0.24 -0.27  0.00  0.40  0.05  0.00  0.0
d9  -0.24 -0.01 -0.61 -0.01  0.17  0.07  0.07  0.00 -0.11  0.72  0.00  0.0
d10  0.46  0.15 -0.03 -0.85  0.21  0.02  0.02  0.00 -0.03  0.06  0.00  0.0
d11  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.0
Sigma
         d0    d1    d2    d3    d4    d5   d6   d7   d8   d9  d10  d11
nyc    2.24  0.00  0.00  0.00  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
lion   0.00  1.63  0.00  0.00  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
love   0.00  0.00  1.36  0.00  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
dog    0.00  0.00  0.00  0.83  0.00  0.00  0.0  0.0  0.0  0.0  0.0  0.0
apple  0.00  0.00  0.00  0.00  0.71  0.00  0.0  0.0  0.0  0.0  0.0  0.0
cat    0.00  0.00  0.00  0.00  0.00  0.56  0.0  0.0  0.0  0.0  0.0  0.0
Sigma without zeroing any dim
[2.24 1.63 1.36 0.83 0.71 0.56]
Reconstructed Term-Document Matrix
[[ 0.67  0.67  0.6   0.67  0.67  0.   -0.    0.    0.    0.6   0.    0.  ]
 [ 0.    0.   -0.    0.    0.    0.    0.    0.83  0.   -0.   -0.    0.  ]
 [ 0.    0.    0.8   0.    0.   -0.    0.    0.    0.74  0.8   0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.83  0.  ]
 [ 0.74  0.74  0.    0.74  0.74  1.   -0.   -0.    0.   -0.   -0.    0.  ]
 [-0.   -0.    0.   -0.   -0.    0.    1.    0.56  0.67  0.    0.56  0.  ]]
Error without reducing dimensions
2.4118514282911836e-16
Sigma after zeroing out dim 5
[2.24 1.63 1.36 0.83 0.71 0.  ]
Error after zeroing out dim 5
0.06622832691855576
Sigma after zeroing out dim 4
[2.24 1.63 1.36 0.83 0.   0.  ]
Error after zeroing out dim 4
0.10640521215974559
Sigma after zeroing out dim 3
[2.24 1.63 1.36 0.   0.   0.  ]
Error after zeroing out dim 3
0.14426065117971226
Sigma after zeroing out dim 2
[2.24 1.63 0.   0.   0.   0.  ]
Error after zeroing out dim 2
0.21538404738257216
Sigma after zeroing out dim 1
[2.24 0.   0.   0.   0.   0.  ]
Error after zeroing out dim 1
0.28856858954287745
[1.         0.93377167 0.89359479 0.85573935 0.78461595 0.71143141]
"""


def lsa_models(vocabulary='cat dog apple lion NYC love'.lower().split(), docs=11, verbosity=0):
    # vocabulary = 'cat dog apple lion NYC love big small bright'.lower().split()
    if isinstance(docs, int):
        docs = get_data('cats_and_dogs_sorted')[:docs]
    tdm, tfidfdm, tfidfer = docs_to_tdm(docs=docs, vocabulary=vocabulary)
    lsa_bow_model = lsa(tdm)  # (tdm - tdm.mean(axis=1)) # SVD fails to converge if you center, like PCA does
    lsa_bow_model['vocabulary'] = tdm.index.values
    lsa_bow_model['docs'] = docs
    err = accuracy_study(verbosity=verbosity, **lsa_bow_model)
    lsa_bow_model['err'] = err
    lsa_bow_model['accuracy'] = list(1. - np.array(err))
    
    lsa_tfidf_model = lsa(tdm=tfidfdm)
    lsa_bow_model['vocabulary'] = tfidfdm.index.values
    lsa_tfidf_model['docs'] = docs
    err = accuracy_study(verbosity=verbosity, **lsa_tfidf_model)
    lsa_tfidf_model['err'] = err
    lsa_tfidf_model['accuracy'] = list(1. - np.array(err))

    return lsa_bow_model, lsa_tfidf_model


if __name__ == '__main__':
    numdocs = 11
    docs = get_data('cats_and_dogs_sorted')[:numdocs]
    vocabulary = sys.argv[1:] or 'cat dog apple lion NYC love'.lower().split()
    lsa_bow_model, lsa_tfidf_model = lsa_models(vocabulary=vocabulary, docs=docs)
    tdm = lsa_bow_model['tdm']
    tfidfdm = lsa_tfidf_model['tdm']
    print(prettify_tdm(tdm=tdm, docs=docs, vocabulary=vocabulary))
    acc = plot_feature_selection(accuracy=lsa_bow_model['accuracy'])
    print("BOW accuracy after multiplying Truncated SVD back together")
    print(acc)
    acc = plot_feature_selection(accuracy=lsa_tfidf_model['accuracy'], title='TF-IDF LSA Model Accuracy')
    print("TF-IDF accuracy after multiplying Truncated SVD back together")
    print(acc)
    plt.legend(['BOW Reconstruction Accuracy', 'TF-IDF Reconstruction Accuracy'])
    yn = input('Would you like to see the accuracy comparison plot? [y]')
    if not yn or yn.lower().startswith('y'):
        plt.show()

