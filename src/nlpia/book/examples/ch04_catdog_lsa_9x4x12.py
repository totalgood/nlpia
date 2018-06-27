"""


"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # noqa
import seaborn  # noqa
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.data.loaders import get_data

pd.options.display.width = 120
pd.options.display.max_columns = 12

corpus = docs = get_data('cats_and_dogs_sorted')[:12]
vocabulary = 'cat dog apple lion nyc love big small bright'.split()
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
bow_pretty = bow_dense.copy()
bow_pretty = bow_pretty[vocabulary]
bow_pretty['text'] = corpus
for col in vocabulary:
    bow_pretty.loc[bow_pretty[col] == 0, col] = ''
print(bow_pretty)
print(bow_pretty.T)
print(tfidf_dense.T)


tdm = bow_dense.T
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

vt = pd.DataFrame(vt, index=['d{}'.format(i) for i in range(len(corpus))])
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


plt.plot(range(len(err)), err)
plt.title('Dimension Reduction Error in Term Frequency')
plt.xlabel('Number of Dimensions Eliminated')
plt.ylabel('Mean Square Error in Term Frequency')
plt.grid(True)
plt.tight_layout()
print(err)
# plt.show()
