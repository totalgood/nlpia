""" Section 2.3 code listings from NLPIA """

import pandas as pd
pd.options.display.max_colwidth = 40  # default: 50
pd.options.display.width = 75  # default: 80
pd.options.display.max_columns = 12  # default: 0


"""
>>> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
>>> sa = SentimentIntensityAnalyzer()
>>> sa.lexicon
{ ...
':(': -1.9,
':)': 2.0,
...
'pls': 0.3,
'plz': 0.3,
...
'great': 3.1,
... }
>>> [(tok, score) for tok, score in sa.lexicon.items()
... if " " in tok]
[("( '}{' )", 1.6),
("can't stand", -2.0),
('fed up', -1.8),
('screwed up', -1.5)]
>>> sa.polarity_scores(text=\
... "Python is very readable and it's great for NLP.")
{'compound': 0.6249, 'neg': 0.0, 'neu': 0.661,
'pos': 0.339}
>>> sa.polarity_scores(text=\
... "Python is not a bad choice for most applications.")
{'compound': 0.431, 'neg': 0.0, 'neu': 0.711,
'pos': 0.289}
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa
sa = SentimentIntensityAnalyzer()
sa.lexicon

"""
>>> corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
... "Horrible! Completely useless. :(",
... "It was OK. Some good and some bad things."]
>>> for doc in corpus:
...     scores = sa.polarity_scores(doc)
...     print('{:+}: {}'.format(scores['compound'], doc))
+0.9428: Absolutely perfect! Love it! :-) :-) :-)
-0.8768: Horrible! Completely useless. :(
+0.3254: It was OK. Some good and some bad things.
"""

corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print('{:+}: {}'.format(scores['compound'], doc))
# +0.9428: Absolutely perfect! Love it! :-) :-) :-)
# -0.8768: Horrible! Completely useless. :(
# +0.3254: It was OK. Some good and some bad things.

"""
>>> from nlpia.data.loaders import get_data
>>> movies = get_data('hutto_movies')
>>> movies.head().round(2)
    sentiment                                     text
id                                                    
1        2.27  The Rock is destined to be the 21st ...
2        3.53  The gorgeously elaborate continuatio...
3       -0.60           Effective but too tepid biopic
4        1.47  If you sometimes like to go to the m...
5        1.73  Emerges as something rare, an issue ...
>>> movies.describe().round(2)
       sentiment
count   10605.00
mean        0.00
std         1.92
min        -3.88
25%        -1.77
50%        -0.08
75%         1.83
max         3.94
"""

from nlpia.data.loaders import get_data  # noqa
movies = get_data('hutto_movies')
movies.head().round(2)
#     sentiment                                     text
# id                                                    
# 1        2.27  The Rock is destined to be the 21st ...
# 2        3.53  The gorgeously elaborate continuatio...
# 3       -0.60           Effective but too tepid biopic
# 4        1.47  If you sometimes like to go to the m...
# 5        1.73  Emerges as something rare, an issue ...
movies.describe().round(2)
#        sentiment
# count   10605.00
# mean        0.00
# std         1.92
# min        -3.88
# 25%        -1.77
# 50%        -0.08
# 75%         1.83
# max         3.94

"""
>>> import pandas as pd
>>> pd.set_option('display.width', 75)
>>> from nltk.tokenize import casual_tokenize
>>> bags_of_words = []
>>> from collections import Counter
>>> for text in movies.text:
        bags_of_words.append(Counter(casual_tokenize(text)))
>>> df_bows = pd.DataFrame.from_records(bags_of_words)
>>> df_bows = df_bows.fillna(0).astype(int)
>>> df_bows.shape
(10605, 20756)
>>> df_bows.head()
   !  "  #  $  %  & ...  zoning  zzzzzzzzz  ½  élan  –  ’
0  0  0  0  0  0  0 ...       0          0  0     0  0  0
1  0  0  0  0  0  0 ...       0          0  0     0  0  0
2  0  0  0  0  0  0 ...       0          0  0     0  0  0
3  0  0  0  0  0  0 ...       0          0  0     0  0  0
4  0  0  0  0  0  0 ...       0          0  0     0  0  0
>>> df_bows.head()[list(bags_of_words[0].keys())]
   The  Rock  is  destined  to  be ...  Van  Damme  or  Steven  Segal  .
0    1     1   1         1   2   1 ...    1      1   1       1      1  1
1    2     0   1         0   0   0 ...    0      0   0       0      0  4
2    0     0   0         0   0   0 ...    0      0   0       0      0  0
3    0     0   1         0   4   0 ...    0      0   0       0      0  1
4    0     0   0         0   0   0 ...    0      0   0       0      0  1
"""

import pandas as pd  # noqa
pd.set_option('display.width', 75)
from nltk.tokenize import casual_tokenize  # noqa
bags_of_words = []
from collections import Counter  # noqa
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
df_bows.shape
# (10605, 20756)
df_bows.head()
#    !  "  #  $  %  & ...  zoning  zzzzzzzzz  ½  élan  –  ’
# 0  0  0  0  0  0  0 ...       0          0  0     0  0  0
# 1  0  0  0  0  0  0 ...       0          0  0     0  0  0
# 2  0  0  0  0  0  0 ...       0          0  0     0  0  0
# 3  0  0  0  0  0  0 ...       0          0  0     0  0  0
# 4  0  0  0  0  0  0 ...       0          0  0     0  0  0
df_bows.head()[list(bags_of_words[0].keys())]
#    The  Rock  is  destined  to  be ...  Van  Damme  or  Steven  Segal  .
# 0    1     1   1         1   2   1 ...    1      1   1       1      1  1
# 1    2     0   1         0   0   0 ...    0      0   0       0      0  4
# 2    0     0   0         0   0   0 ...    0      0   0       0      0  0
# 3    0     0   1         0   4   0 ...    0      0   0       0      0  1
# 4    0     0   0         0   0   0 ...    0      0   0       0      0  1

"""
>>> from sklearn.naive_bayes import MultinomialNB
>>> nb = MultinomialNB()
>>> nb = nb.fit(df_bows, movies.sentiment > 0)
>>> movies['predicted_sentiment'] = nb.predict(df_bows) * 8 - 4
>>> movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
>>> movies.error.mean().round(1)
2.4
>>> movies['predicted_ispos'] = (movies.predicted_sentiment > 0).astype(int)
>>> movies['sentiment predicted_sentiment sentiment_ispositive predicted_ispos'
...        .split()].head(8)
    sentiment  predicted_sentiment  sentiment_ispositive  predicted_ispos
id                                                                       
1    2.266667                    4                     1                1
2    3.533333                    4                     1                1
3   -0.600000                   -4                     0                0
4    1.466667                    4                     1                1
5    1.733333                    4                     1                1
6    2.533333                    4                     1                1
7    2.466667                    4                     1                1
8    1.266667                   -4                     1                0
>>> (movies.predicted_ispositive ==
...  movies.sentiment_ispositive).sum() / len(movies)
"""

from sklearn.naive_bayes import MultinomialNB  # noqa
nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
movies['predicted_sentiment'] = nb.predict(df_bows) * 8 - 4
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
movies.error.mean().round(1)
# 2.4
movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispos'] = (movies.predicted_sentiment > 0).astype(int)
movies['sentiment predicted_sentiment sentiment_ispositive predicted_ispos'
       .split()].head(8)
#     sentiment  predicted_sentiment  sentiment_ispositive  predicted_ispos
# id                                                                       
# 1    2.266667                    4                     1                1
# 2    3.533333                    4                     1                1
# 3   -0.600000                   -4                     0                0
# 4    1.466667                    4                     1                1
# 5    1.733333                    4                     1                1
# 6    2.533333                    4                     1                1
# 7    2.466667                    4                     1                1
# 8    1.266667                   -4                     1                0


"""
>>> products = get_data('hutto_products')
>>> bags_of_words = []
>>> for text in products.text:
...     bags_of_words.append(Counter(casual_tokenize(text)))
>>> df_product_bows = pd.DataFrame.from_records(bags_of_words)
>>> df_product_bows = df_product_bows.fillna(0).astype(int)
>>> df_all_bows = df_bows.append(df_product_bows)
>>> df_all_bows.columns
Index(['!', '"', '#', '#38', '$', '%', '&', ''', '(', '(8',
      ...
      'zoomed', 'zooming', 'zooms', 'zx', 'zzzzzzzzz', '~', '½', 'élan',
      '–', '’'],
      dtype='object', length=23302)
>>> df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]
"""
products = get_data('hutto_products')
bags_of_words = []
for text in products.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_product_bows = pd.DataFrame.from_records(bags_of_words)
df_product_bows = df_product_bows.fillna(0).astype(int)
df_all_bows = df_bows.append(df_product_bows)


"""
>>> df_all_bows.columns.values
array(['!', '"', '#', ..., 'élan', '–', '’'], dtype=object)
>>> df_all_bows.columns
Index(['!', '"', '#', '#38', '$', '%', '&', ''', '(', '(8',
      ...
      'zoomed', 'zooming', 'zooms', 'zx', 'zzzzzzzzz', '~', '½', 'élan',
      '–', '’'],
      dtype='object', length=23302)
>>> df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]
>>> df_product_bows.shape
(3546, 20756)
>>> df_bows.shape
(10605, 20756)
>>> products['ispos'] = (products.sentiment > 0).astype(int)
>>> products['pred'] = nb.predict(df_product_bows.values).astype(int)
"""
df_all_bows.columns.values
#  array(['!', '"', '#', ..., 'élan', '–', '’'], dtype=object)
df_all_bows.columns
# Index(['!', '"', '#', '#38', '$', '%', '&', ''', '(', '(8',
#       ...
#       'zoomed', 'zooming', 'zooms', 'zx', 'zzzzzzzzz', '~', '½', 'élan',
#       '–', '’'],
#       dtype='object', length=23302)
df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]
df_product_bows.shape
# (3546, 20756)
df_product_bows = df_product_bows.fillna(0).astype(int)
df_bows.shape
# (10605, 20756)
products['ispos'] = (products.sentiment > 0).astype(int)
products['pred'] = nb.predict(df_product_bows.values).astype(int)


"""
>>> products.head()
    id  sentiment                                     text  ispos  pred
0  1_1      -0.90  troubleshooting ad-2500 and ad-2600 ...      0     0
1  1_2      -0.15  repost from january 13, 2004 with a ...      0     0
2  1_3      -0.20  does your apex dvd player only play ...      0     0
3  1_4      -0.10  or does it play audio and video but ...      0     0
4  1_5      -0.50  before you try to return the player ...      0     0
>>> (products.pred == products.ispos).sum() / len(products)
0.5572476029328821
"""
products.head()
#     id  sentiment                                     text  ispos  pred
# 0  1_1      -0.90  troubleshooting ad-2500 and ad-2600 ...      0     0
# 1  1_2      -0.15  repost from january 13, 2004 with a ...      0     0
# 2  1_3      -0.20  does your apex dvd player only play ...      0     0
# 3  1_4      -0.10  or does it play audio and video but ...      0     0
# 4  1_5      -0.50  before you try to return the player ...      0     0
(products.pred == products.ispos).sum() / len(products)
# 0.5572476029328821
