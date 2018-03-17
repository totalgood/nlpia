from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases() # noqa
from builtins import *  # noqa

import os
import gc

import json
import pandas as pd
import gzip
from tqdm import tqdm

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary

from nlpia.data import BIGDATA_PATH, read_csv
from nlpia.gensim_utils import TweetCorpus

KEEP_N = 300000   # max vocab size
NO_BELOW = 5      # min DF (count)
NO_ABOVE = .99    # max DF (fraction)

# Only 5 of these tokens are saved for a no_below=2 filter: PyCons NLPS #PyCon2016 #NaturalLanguageProcessing #naturallanguageprocessing
cased_tokens = 'PyConOpenSpaces PyCon PyCon2017 PyCon2018 PyCon2016 PyCon2015 OpenSpace PyconTutorial'.split()
cased_tokens += 'NLP NaturalLanguageProcessing NLPInAction NaturalLanguageProcessingInAction NLPIA Twote Twip'.split()
cased_tokens += [s + 's' for s in cased_tokens]
allcase_tokens = cased_tokens + [s.lower() for s in cased_tokens]
allcase_tokens += [s.title() for s in cased_tokens]
allcase_tokens += [s.upper() for s in cased_tokens]
KEEP_TOKENS = allcase_tokens + ['#' + s for s in allcase_tokens]

# takes 15 minutes and 10GB of RAM for 500k tweets if you keep all 20M unique tokens/names URLs
vocab_path = os.path.join(BIGDATA_PATH, 'vocab.gensim')
if os.path.isfile(vocab_path):
    print('Loading vocab: {} ...'.format(vocab_path))
    vocab = Dictionary.load(vocab_path)
    print(' len(vocab) loaded: {}'.format(len(vocab.dfs)))
else:
    tokens_path = os.path.join(BIGDATA_PATH, 'tweets.txt.gz')
    if not os.path.isfile(tokens_path):
        tweets_path = os.path.join(BIGDATA_PATH, 'tweets.csv.gz')
        print('Loading tweets: {} ...'.format(tweets_path))
        tweets = read_csv(tweets_path)
        tweets = pd.np.array(tweets.text.apply(eval).apply(bytes.decode).str.split())
        with gzip.open(tokens_path, 'wb') as f:
            for tokens in tqdm(tweets):
                f.write((b' '.join(tokens) + b'\n'))
    # tweets['text'] = tweets.text.apply(lambda s: eval(s).decode('utf-8'))
    # tweets['user'] = tweets.user.apply(lambda s: eval(s).decode('utf-8'))
    # tweets.to_csv('tweets.csv.gz', compression='gzip')
    print('Computing vocab from tweets in {} ...'.format(tokens_path))
    corpus = TweetCorpus()
    corpus.input = gzip.open(tokens_path, 'rb')
    vocab = Dictionary(corpus, prune_at=20000000)
    vocab.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_tokens=set(KEEP_TOKENS))

vocab.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N, keep_tokens=set(KEEP_TOKENS))
print(' len(vocab) after filtering: {}'.format(len(vocab.dfs)))


# no time at all, just a bookeeping step, doesn't actually compute anything
tfidf = TfidfModel(id2word=vocab, dictionary=vocab)
tfidf.save(os.path.join(BIGDATA_PATH, 'tfidf{}.pkl'.format(len(vocab.dfs))))

tweets = [vocab.doc2bow(tw) for tw in tweets]
json.dump(tweets, gzip.open(os.path.join(BIGDATA_PATH, 'tweet_bows.json.gz'), 'w'))

gc.collect()

# LSA is more useful name than LSA
lsa = LsiModel(tfidf[tweets], num_topics=200, id2word=vocab, one_pass=False, extra_samples=150, power_iters=3)

# these models can be big
lsa.save(os.path.join(BIGDATA_PATH, 'lsa_tweets'))
