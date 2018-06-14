"""
Term-Document Matrix (BOW vectors) from Stanford Information Retrieval textbook, 2009 Cambridge University Press

>>> tdm = pd.DataFrame([list(t) for t in '101000 010000 110000 100110 000101'.split()],
...                    columns=['d{}'.format(i) for i in range(6)],
...                    index='ship boat ocean voyage trip'.split()).astype(int)
>>> tdm
        d0  d1  d2  d3  d4  d5
ship     1   0   1   0   0   0
boat     0   1   0   0   0   0
ocean    1   1   0   0   0   0
voyage   1   0   0   1   1   0
trip     0   0   0   1   0   1

>>> u, s, vt = np.linalg.svd(tdm)

>>> u = pd.DataFrame(u, index=tdm.index)
>>> u.round(2)
           0     1     2     3     4
ship    0.44 -0.30 -0.57  0.58 -0.25
boat    0.13 -0.33  0.59  0.00 -0.73
ocean   0.48 -0.51  0.37  0.00  0.61
voyage  0.70  0.35 -0.15 -0.58 -0.16
trip    0.26  0.65  0.41  0.58  0.09

>>> smat = np.zeros(tdm.shape)
>>> for i, value in enumerate(s):
...     smat[i, i] = value
>>> smat = pd.DataFrame(smat, columns=tdm.columns, index=tdm.index)
>>> smat.round(2)
           0     1     2    3     4
ship    2.16  0.00  0.00  0.0  0.00
boat    0.00  1.59  0.00  0.0  0.00
ocean   0.00  0.00  1.28  0.0  0.00
voyage  0.00  0.00  0.00  1.0  0.00
trip    0.00  0.00  0.00  0.0  0.39

>>> vt = pd.DataFrame(vt, index=[f'd{i}' for i in range(6)])
>>> vt.round(2)
       0     1     2     3     4     5
d0  0.75  0.28  0.20  0.45  0.33  0.12
d1 -0.29 -0.53 -0.19  0.63  0.22  0.41
d2 -0.28  0.75 -0.45  0.20 -0.12  0.33
d3 -0.00  0.00  0.58  0.00 -0.58  0.58
d4  0.53 -0.29 -0.63 -0.19 -0.41  0.22
d5  0.00 -0.00 -0.00 -0.58  0.58  0.58

Reconstruct the original term-document matrix.
The sum of the squares of the error is 0.

>>> tdm_prime = u.values @ smat.values @ vt.values
>>> tdm_prime.round(2)
array([[ 1., -0.,  1., -0., -0., -0.],
       [-0.,  1., -0., -0.,  0., -0.],
       [ 1.,  1., -0.,  0., -0.,  0.],
       [ 1.,  0., -0.,  1.,  1.,  0.],
       [-0.,  0., -0.,  1.,  0.,  1.]])
>>> np.sqrt(((tdm_prime - tdm).values.flatten() ** 2).sum())
2.3481474529927113e-15

>>> smat2 = smat.copy()
>>> for i in range(2, len(s)):
...     smat2.iloc[i, i] = 0
>>> smat2.round(2)
          d0    d1   d2   d3   d4   d5
ship    2.16  0.00  0.0  0.0  0.0  0.0
boat    0.00  1.59  0.0  0.0  0.0  0.0
ocean   0.00  0.00  0.0  0.0  0.0  0.0
voyage  0.00  0.00  0.0  0.0  0.0  0.0
trip    0.00  0.00  0.0  0.0  0.0  0.0




"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # noqa
import seaborn  # noqa
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.data.loaders import get_data


VOCABULARY = 'cat dog apple lion NYC love big small'.lower().split()
DOCS = get_data('cats_and_dogs_sorted')[:12]


def docs_to_tdm(docs=DOCS, vocabulary=VOCABULARY):
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
    print(tfidf_dense.T)
    return bow_dense.T, tfidf_dense.T, tfidfer


def prettify_tdm(tdm, docs, vocabulary):
    bow_pretty = tdm.T.copy()
    bow_pretty = bow_pretty[vocabulary]
    bow_pretty['text'] = docs
    for col in vocabulary:
        bow_pretty.loc[bow_pretty[col] == 0, col] = ''
    return bow_pretty


def lsa(tdm):
    print(tdm)
    #         d0  d1  d2  d3  d4  d5
    # ship     1   0   1   0   0   0
    # boat     0   1   0   0   0   0
    # ocean    1   1   0   0   0   0
    # voyage   1   0   0   1   1   0
    # trip     0   0   0   1   0   1

    u, s, vt = np.linalg.svd(tdm)

    u = pd.DataFrame(u, index=tdm.index)
    print('U')
    print(u.round(2))
    #            0     1     2     3     4
    # ship    0.44 -0.30 -0.57  0.58 -0.25
    # boat    0.13 -0.33  0.59  0.00 -0.73
    # ocean   0.48 -0.51  0.37  0.00  0.61
    # voyage  0.70  0.35 -0.15 -0.58 -0.16
    # trip    0.26  0.65  0.41  0.58  0.09

    smat = np.zeros(tdm.shape)
    for i, value in enumerate(s):
        smat[i, i] = value
    smat = pd.DataFrame(smat, columns=tdm.columns, index=tdm.index)
    print('Sigma')
    print(smat.round(2))
    #            0     1     2    3     4
    # ship    2.16  0.00  0.00  0.0  0.00
    # boat    0.00  1.59  0.00  0.0  0.00
    # ocean   0.00  0.00  1.28  0.0  0.00
    # voyage  0.00  0.00  0.00  1.0  0.00
    # trip    0.00  0.00  0.00  0.0  0.39

    vt = pd.DataFrame(vt, index=['d{}'.format(i) for i in range(len(vt))])
    print('VT')
    print(vt.round(2))
    #        0     1     2     3     4     5
    # d0  0.75  0.28  0.20  0.45  0.33  0.12
    # d1 -0.29 -0.53 -0.19  0.63  0.22  0.41
    # d2 -0.28  0.75 -0.45  0.20 -0.12  0.33
    # d3 -0.00  0.00  0.58  0.00 -0.58  0.58
    # d4  0.53 -0.29 -0.63 -0.19 -0.41  0.22
    # d5  0.00 -0.00 -0.00 -0.58  0.58  0.58

    # Reconstruct the original term-document matrix.
    # The sum of the squares of the error is 0.

    print('Sigma without zeroing any dim')
    print(np.diag(smat.round(2)))
    tdm_prime = u.values.dot(smat.values).dot(vt.values)
    print('Reconstructed Term-Document Matrix')
    print(tdm_prime.round(2))
    # array([[ 1., -0.,  1., -0., -0., -0.],
    #        [-0.,  1., -0., -0.,  0., -0.],
    #        [ 1.,  1., -0.,  0., -0.,  0.],
    #        [ 1.,  0., -0.,  1.,  1.,  0.],
    #        [-0.,  0., -0.,  1.,  0.,  1.]])
    err = [np.sqrt(((tdm_prime - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))]
    print('Error without reducing dimensions')
    print(err[-1])
    2.3481474529927113e-15

    smat2 = smat.copy()
    for numdim in range(len(s) - 1, 0, -1):
        smat2.iloc[numdim, numdim] = 0
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
        print('Error after zeroing out dim {}'.format(numdim))
        print(err[-1])
        # 1.6677932876555255
    return {'U': u, 'S': s, 'VT': vt, 'Accuracy': 1. - np.array(err)}


def plot_feature_selection(accuracy):
    # accuracy = topic_model['Accuracy']
    plt.plot(range(len(accuracy)), accuracy)
    plt.title('LSA Model Accuracy')
    plt.xlabel('Number of Dimensions Eliminated')
    plt.ylabel('Reconstruction Accuracy')
    plt.grid(True)
    plt.tight_layout()
    print(accuracy)
    plt.show()


if __name__ == '__main__':
    vocabulary = 'cat dog apple lion NYC love big small'.lower().split()
    docs = get_data('cats_and_dogs_sorted')[:12]
    tdm, tfidfdm, tfidfer = docs_to_tdm(docs=docs, vocabulary=vocabulary)
    print(prettify_tdm(tdm, docs=docs, vocabulary=vocabulary))
    model = lsa(tdm=tdm)
    plot_feature_selection(accuracy=model['Accuracy'])
