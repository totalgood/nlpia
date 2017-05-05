#!/usr/bin/env python3
"""model_isbot

Run nlpia.data.download() to download GBs of models like W2V and the LSAmodel used here
"""
from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases() # noqa
from builtins import *  # noqa

import os

import pandas as pd
from tqdm import tqdm
import gc

from gensim.models import LsiModel, TfidfModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
# from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from nlpia.constants import BIGDATA_PATH
from nlpia.data import read_csv

from nlpia.embedders import positive_distances

# adjust this downward
np = pd.np


def train_lda(training_size=2000, metric='cosine'):
    tweets = read_csv(os.path.join(BIGDATA_PATH, 'tweets.csv.gz'))
    tweets = tweets[tweets.isbot >= 0]

    # labels3 = tweets.isbot.apply(lambda x: int(x * 3))
    labels = tweets.isbot.apply(lambda x: int(x * 2))

    lsa = LsiModel.load(os.path.join(BIGDATA_PATH, 'lsa_tweets_5589798_2003588x200.pkl'))
    tfidf = TfidfModel(id2word=lsa.id2word, dictionary=lsa.id2word)
    bows = np.array([lsa.id2word.doc2bow(txt.split()) for txt in tweets.text])
    # tfidfs = tfidf[bows]

    X = pd.DataFrame([pd.Series(dict(v)) for v in tqdm(lsa[tfidf[bows]], total=len(bows))], index=tweets.index)
    mask = ~X.isnull().any(axis=1)
    mask.index = tweets.index
    X = X[mask]
    y = tweets.isbot[mask]
    labels = labels[mask]
    # labels3 = labels3[mask]

    test_size = 1.0 - training_size if training_size < 1 else float(len(X) - training_size) / len(X)
    Xindex, Xindex_test, yindex, yindex_test = train_test_split(X.index.values, y.index.values, test_size=test_size)
    X, Xtest, y, ytest = X.loc[Xindex], X.loc[Xindex_test], y.loc[yindex], y.loc[yindex_test]
    labels_test = labels.loc[yindex_test]
    labels = labels.loc[yindex]

    lda = LDA('lsqr', 'auto', n_components=3)
    print(cross_val_score(lda, Xtest, labels_test, cv=7))

    lda = LDA('lsqr', 'auto', n_components=3)
    lda = lda.fit(X.values, labels.values)
    y_lda = lda.predict(Xtest)
    print(mean_squared_error(y_lda, ytest))

    df_test = pd.DataFrame(lda.predict(Xtest), index=Xtest.index, columns=['predict'])
    df_test['truth'] = labels_test
    return lda, df_test


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
    
    # labels_test = labels.loc[yindex_test]
    labels = labels.loc[yindex]

    tsne = TSNE(metric='precomputed', n_components=n_components, angle=angle, perplexity=perplexity)
    tsne = tsne.fit(positive_distances(X.values, metric=metric))

    return tsne, X, Xtest, y, ytest

# lda.save('lda')

# svr_lin = SVR(kernel='linear', C=1e3)
# y_lin = svr_lin.fit(X, y).predict(Xtest)
# print(mean_squared_error(y_lin, ytest))
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# y_rbf = svr_rbf.fit(X, y).predict(Xtest)
# lda.save('svr_rbf')

# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
# y_poly = svr_poly.fit(X, y).predict(Xtest)
