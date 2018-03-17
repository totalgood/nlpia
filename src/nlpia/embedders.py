#!/usr/bin/env python3
"""model_poly_tsne

Run nlpia.data.download() to download GBs of models like W2V and the LSAmodel used here

Computes a TSNE embedding for the tweet LSA model and then fit a 2nd degree polynomial to that embedding.
"""

import os
import gc

import pandas as pd
from tqdm import tqdm

from gensim.models import LsiModel, TfidfModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.svm import SVR
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from nlpia.constants import BIGDATA_PATH
from nlpia.data import read_csv

import sklearn.metrics.pairwise

np = pd.np


def positive_projection(x, y, max_norm=1.0):
    proj = max_norm - float(np.dot(x, y))
    if proj < 1e-15:
        print(x, y, proj)
    return max(proj, 0.0) ** 0.5


def positive_distances(X, metric='cosine'):
    X = X.values if (hasattr(X, 'values') and not callable(X.values)) else X
    metric = getattr(sklearn.metrics.pairwise, metric + '_distances') if isinstance(metric, basestring) else metric
    distances = metric(X)
    distances[distances < 0] = 0.0
    return distances


def bent_distances(X, y, weight=1.0, metric='cosine'):
    y = np.array(y).reshape((len(X), 1))
    distances = positive_distances(X, metric=metric)
    distances += weight * sklearn.metrics.pairwise.euclidean_distances(np.matrix(y).T)
    return distances


def train_tsne(training_size=2000, metric='cosine', n_components=3, perplexity=100, angle=.12):
    # adjust this downward to see it it affects accuracy
    np = pd.np

    tweets = read_csv(os.path.join(BIGDATA_PATH, 'tweets.csv.gz'))
    tweets = tweets[tweets.isbot >= 0]
    gc.collect()  # reclaim RAM released above

    # labels3 = tweets.isbot.apply(lambda x: int(x * 3))
    labels = tweets.isbot.apply(lambda x: int(x * 2))

    lsa = LsiModel.load(os.path.join(BIGDATA_PATH, 'lsa_tweets_5589798_2003588x200.pkl'))
    tfidf = TfidfModel(id2word=lsa.id2word, dictionary=lsa.id2word)
    bows = np.array([lsa.id2word.doc2bow(txt.split()) for txt in tweets.text])
    # tfidfs = tfidf[bows]

    X = pd.DataFrame([pd.Series(dict(v)) for v in tqdm(lsa[tfidf[bows]], total=len(bows))], index=tweets.index)

    mask = ~X.isnull().any(axis=1)
    mask.index = tweets.index
    # >>> sum(~mask)
    # 99
    # >>> tweets.loc[mask.argmin()]
    # isbot                 0.17
    # strict                  13
    # user      b'CrisParanoid:'
    # text         b'#sad again'
    # Name: 571, dtype: object

    X = X[mask]
    y = tweets.isbot[mask]
    labels = labels[mask]

    test_size = 1.0 - training_size if training_size < 1 else float(len(X) - training_size) / len(X)
    Xindex, Xindex_test, yindex, yindex_test = train_test_split(X.index.values, y.index.values, test_size=test_size)
    X, Xtest, y, ytest = X.loc[Xindex], X.loc[Xindex_test], y.loc[yindex], y.loc[yindex_test]
    labels_test = labels.loc[yindex_test]
    labels = labels.loc[yindex]

    tsne = TSNE(metric='precomputed', n_components=n_components, angle=angle, perplexity=perplexity)
    tsne = tsne.fit(positive_distances(X.values, metric=metric))

    return tsne, X, Xtest, y, ytest


def embedding_correlation(Xtest, ytest):
    pass


def plot_embedding(tsne, labels, index=None):
    labels = labels.values if (hasattr(labels, 'values') and not callable(labels.values)) else labels
    colors = np.array(list('gr'))[labels]
    df = pd.DataFrame(tsne.embedding_, columns=list('xy'), index=index)
    return df.plot(kind='scatter', x='x', y='y', c=colors)
