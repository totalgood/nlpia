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

# adjust this downward
NUM_FOR_TRAINING = 1000
np = pd.np

tweets = read_csv(os.path.join(BIGDATA_PATH, 'tweets.csv.gz'))
tweets = tweets[tweets.isbot >= 0]
gc.collect()  # reclaim RAM released above

# labels3 = tweets.isbot.apply(lambda x: int(x * 3))
labels2 = tweets.isbot.apply(lambda x: int(x * 2))

lsa = LsiModel.load(os.path.join(BIGDATA_PATH, 'lsa_tweets_5589798_2003588x200.pkl'))
tfidf = TfidfModel(id2word=lsa.id2word, dictionary=lsa.id2word)
bows = np.array([lsa.id2word.doc2bow(txt.split()) for txt in tweets.text])
tfidfs = tfidf[bows]

X = pd.DataFrame([pd.Series(dict(v)) for v in tqdm(lsa[tfidf[bows]], total=len(bows))], index=tweets.index)

mask = ~X.isnull().any(axis=1)
mask.index = tweets.index
X = X[mask]
y = tweets.isbot[mask]
labels2 = labels2[mask]

Xindex, Xindex_test, yindex, yindex_test = train_test_split(X.index.values, y.index.values, test_size=.25)
X, Xtest, y, ytest = X.loc[Xindex], X.loc[Xindex_test], y.loc[yindex], y.loc[yindex_test]
labels2_test = labels2.loc[yindex_test]
labels2 = labels2.loc[yindex]

tsne = TSNE()
tsne.fit(X)
