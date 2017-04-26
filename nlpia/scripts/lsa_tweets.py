import os
import gc

import json
import pandas as pd
import gzip

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary

from nlpia.data import BIGDATA_PATH, read_csv

KEEP_N = 300000  # max vocab size
NO_BELOW = 5      # min DF (count)
NO_ABOVE = .7     # max DF (fraction)

# Only 5 of these tokens are saved for a no_below=2 filter: PyCons NLPS #PyCon2016 #NaturalLanguageProcessing #naturallanguageprocessing
cased_tokens = 'PyConOpenSpaces PyCon PyCon2017 PyCon2018 PyCon2016 PyCon2015 OpenSpace PyconTutorial'.split()
cased_tokens += 'NLP NaturalLanguageProcessing NLPInAction NaturalLanguageProcessingInAction NLPIA Twote Twip'.split()
cased_tokens += [s + 's' for s in cased_tokens]
cased_tokens += 'TotalGood TotalGoods HobsonLane Hob Hobs TotalGood.com www.TotalGood.com http://www.TotalGood.com https://www.TotalGood.com'.split()
allcase_tokens = cased_tokens + [s.lower() for s in cased_tokens]
allcase_tokens += [s.title() for s in cased_tokens]
allcase_tokens += [s.upper() for s in cased_tokens]
KEEP_TOKENS = allcase_tokens + ['#' + s for s in allcase_tokens]

# takes 15 minutes and 10GB of RAM for 500k tweets if you keep all 20M unique tokens/names URLs
vocab_path = os.path.join(BIGDATA_PATH, 'vocab939370.pkl')
if os.path.isfile(vocab_path):
    print('Loading vocab: {} ...'.format(vocab_path))
    vocab = Dictionary.load(vocab_path)
    print(' len(vocab) loaded: {}'.format(len(vocab.dfs)))
else:
    tweets_path = os.path.join(BIGDATA_PATH, 'tweets.csv.gz')
    print('Loading tweets: {} ...'.format(tweets_path))
    tweets = read_csv(tweets_path)
    tweets = pd.np.array(tweets.text.str.split())
    with gzip.open(os.path.join(BIGDATA_PATH, 'tweets.txt.gz'), 'w') as f:
        for tokens in tweets:
            f.write((' '.join(tokens) + '\n').encode('utf-8'))
    # tweets['text'] = tweets.text.apply(lambda s: eval(s).decode('utf-8'))
    # tweets['user'] = tweets.user.apply(lambda s: eval(s).decode('utf-8'))
    # tweets.to_csv('tweets.csv.gz', compression='gzip')
    print('Computing vocab from {} tweets...'.format(len(tweets)))
    vocab = Dictionary(tweets, no_below=NO_BELOW, no_above=NO_ABOVE, keep_tokens=set(KEEP_TOKENS))

vocab.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N, keep_tokens=set(KEEP_TOKENS))
print(' len(vocab) after filtering: {}'.format(len(vocab.dfs)))


# no time at all, just a bookeeping step, doesn't actually compute anything
tfidf = TfidfModel(id2word=vocab, dictionary=vocab)
tfidf.save(os.path.join(BIGDATA_PATH, 'tfidf{}.pkl'.format(len(vocab.dfs))))

tweets = [vocab.doc2bow(tw) for tw in tweets]
json.dump(tweets, gzip.open(os.path.join(BIGDATA_PATH, 'tweet_bows.json.gz'), 'w'))

gc.collect()

# LSA is more useful name than LSA
lsa = LsiModel(tfidf[tweets], num_topics=200, id2word=vocab, extra_samples=100, power_iters=2)

# these models can be big
lsa.save(os.path.join(BIGDATA_PATH, 'lsa_tweets'))
