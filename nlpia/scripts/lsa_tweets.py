import os

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary

import pandas as pd

from nlpia.data import BIGDATA_PATH, read_csv


tweets = read_csv(os.path.join(BIGDATA_PATH, 'tweets.csv.gz'))
tweets = pd.np.array(tweets.txt.str.split())

# takes 15 minutes and 15GB of RAM for 500k tweets
vocab = Dictionary(tweets)

# no time at all, just a bookeeping step, doesn't actually compute anything
tfidf = TfidfModel(id2word=vocab, dictionary=vocab)

# redundant split(), but that's the easy part
bows = pd.Series(vocab.doc2bow(tw) for tw in tweets)

# LSA is more useful name than LSA
lsa = LsiModel(tfidf[bows], num_topics=200, id2word=vocab, extra_samples=100, power_iters=2)

# these models can be big
lsa.save(os.path.join(BIGDATA_PATH, 'lsa_tweets'))
