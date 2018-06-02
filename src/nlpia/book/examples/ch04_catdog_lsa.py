from nlpia.data.loaders import get_data
from nltk.tokenize import casual_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

# import nltk
# nltk.download('wordnet')  # noqa
# from nltk.stem.wordnet import WordNetLemmatizer

corpus = get_data('cats_and_dogs')

STOPWORDS = ('a an and or the do are with from for of on in by if at to into them' +
             'it its it\'s that than our you your ? , . !').split()
SYNONYMS = dict(zip('wolv people person women woman man human he  we  her she him his hers'.split(),
                    'wolf her    her    her   her   her her   her her her her her her her'.split()))
SYNONYMS.update(dict(zip('ate pat smarter have had isn\'t hasn\'t no  got get become been was were wa be sat seat sit'.split(),
                         'eat pet smart   has  has not    not     not has has is     is   is  is   is is sit sit  sit'.split())))
SYNONYMS.update(dict(zip('i me my mine our ours catbird bird birds birder tortoise turtle turtles turtle\'s'.split(),
                         'i i  i  i    i   i    bird    bird birds bird   turtle   turtle turtle  turtle'.split())))

docs = [doc.lower() for doc in corpus]
docs = [casual_tokenize(doc) for doc in docs]
docs = [[SYNONYMS.get(w, w) for w in words if w not in STOPWORDS] for words in docs]
stemmer = PorterStemmer()
docs = [[stemmer.stem(w) for w in words if w not in STOPWORDS] for words in docs]
docs = [[SYNONYMS.get(w, w) for w in words if w not in STOPWORDS] for words in docs]
docs = [' '.join(w for w in words if w not in STOPWORDS) for words in docs]


tfidfer = TfidfVectorizer(min_df=2, max_df=.6, stop_wrods=None, token_pattern=r'(?u)\b\w+\b')
tfidf_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
id_words = [(i, w) for (w, i) in tfidfer.vocabulary_.items()]
tfidf_dense.columns = list(zip(*sorted(id_words)))[1]
"""
>>> tfidf_dense.shape
(200, 170)
"""

pd.options.display.width = 150
pd.options.display.max_columns = 16
tfidf_pretty = tfidf_dense.copy()
tfidf_pretty = tfidf_pretty[['bike', 'cat', 'car', 'chase', 'dog', 'hat', 'i']].head(10).round(1)
tfidf_pretty[tfidf_pretty == 0] = ''
tfidf_pretty['...'] = ''
tfidf_pretty['text'] = corpus[:10]
tfidf_pretty
"""
>>> pd.options.display.width = 150
>>> pd.options.display.max_columns = 16
>>> tfidf_pretty = tfidf_dense.copy()
>>> tfidf_pretty = tfidf_pretty[['bike', 'cat', 'car', 'chase', 'dog', 'hat', 'i']].head(10).round(1)
>>> tfidf_pretty[tfidf_pretty == 0] = ''
>>> tfidf_pretty['...'] = ''
>>> tfidf_pretty['text'] = corpus[:10]
>>> tfidf_pretty
   dog  cat bear  pet  hat bike chase bark meow ...                                         text
0       0.4            0.9                                                    The Cat in the Hat
1       0.5                                                                   The cat ate a rat.
2       0.6                       0.8                           The cat chased my laser pointer.
3  0.3                      0.6   0.4  0.6               A dog chased my bike and barked loudly.
4  0.4                                 0.6                           I ran from the barking dog.
5  0.4                            0.5  0.6                        A dog chased the car, barking.
6       0.5                                 0.9                                   The Cat's Meow
7       0.3       0.4                       0.5      The cat meowed so I pet it until it purred.
8       0.5                                 0.9                 A cat meowed on the hot tin roof
9  0.5  0.4                                                      Cats and dogs playing together.
"""

tfidf_zeros = tfidf_dense.T.sum()[tfidf_dense.T.sum() == 0]
tfidf_zeros
"""
>>> tfidf_zeros = tfidf_dense.T.sum()[tfidf_dense.T.sum() == 0]
>>> tfidf_zeros
199    0.0
"""


[corpus[i] for i in tfidf_zeros.index]
"""
>>> [corpus[i] for i in tfidf_zeros.index]
[]

# ['I flew a kite.', 'He froze.']
"""

from sklearn.decomposition import PCA
pcaer = PCA(n_components=3)

doc_topic_vectors = pd.DataFrame(pcaer.fit_transform(tfidf_dense.values), columns='A B C'.split())
doc_topic_vectors.round(1)
#        A    B    C
# 0    0.0  0.4 -0.1
# 1    0.0  0.5 -0.2
# 2    0.5 -0.2 -0.0
# ...
# 197 -0.2 -0.2 -0.0
# 198 -0.2 -0.2 -0.0
# 199  0.4 -0.2 -0.0
"""
doc_topic_vectors.round(2)
        A     B     C
0    0.01  0.38 -0.14
1    0.03  0.48 -0.17
2    0.49 -0.21 -0.01
3   -0.14  0.03  0.08
...
"""


"""
>>> tfidf_similarity = []
... topic_similarity = []
... for i in range(10):
...     topic_similarity.append((doc_topic_vectors.iloc[i] * doc_topic_vectors.iloc[i+1]).sum())
...     tfidf_similarity.append((tfidf_dense.iloc[i] * tfidf_dense.iloc[i+1]).sum())
>>> tfidf_pretty['tfidf_similar'] = tfidf_similarity
>>> tfidf_pretty['topic_similar'] = topic_similarity
>>> tfidf_pretty
  bike  cat  car chase  dog  hat    i ...                                     text  tfidf_similar  topic_similar
0       0.4                  0.9                                The Cat in the Hat       0.217066       0.208128
1       0.5                                                     The cat ate a rat.       0.000000      -0.085262
2  0.5                            0.6                         I rode my bike home.       0.000000      -0.075669
3            0.5                                         The car is in the garage.       0.267063       0.045623
4            0.5   0.5  0.4                               Dogs like to chase cars.       0.620088       0.214047
5                       0.4                            The post man likes our dog.       0.316328       0.164364
6                       0.3                  He refused to sleep in the dog house.       0.000000      -0.057364
7       0.6        0.8                                     The cat chased a mouse.       0.000000      -0.004849
8                       0.4       0.4           I was in the dog house last night.       0.488741       0.025706
9       0.3             0.3                The cat steered clear of the dog house.       0.000000       0.011857
"""


def tokenize(text, corpus=tfidf_dense):
    docs = [text.lower()]
    docs = [casual_tokenize(doc) for doc in docs]
    docs = [[SYNONYMS.get(w, w) for w in words if w not in STOPWORDS] for words in docs]
    stemmer = PorterStemmer()
    docs = [[stemmer.stem(w) for w in words if w not in STOPWORDS] for words in docs]
    docs = [[SYNONYMS.get(w, w) for w in words if w not in STOPWORDS] for words in docs]
    docs = [' '.join(w for w in words if w not in STOPWORDS) for words in docs]
    stems = [w for w in docs[0].split() if w in corpus.columns]
    return stems


def tfidf_search(text, corpus=tfidf_dense, corpus_text=corpus):
    """ search for the most relevant document """
    tokens = tokenize(text)
    tfidf_vector_query = pd.np.array(tfidfer.transform([' '.join(tokens)]).todense())[0]
    query_series = pd.Series(tfidf_vector_query, index=corpus.columns)

    return corpus_text[query_series.dot(corpus.T).values.argmax()]


def topic_search(text, corpus=doc_topic_vectors, pcaer=pcaer, corpus_text=corpus):
    """ search for the most relevant document """
    tokens = tokenize(text)
    tfidf_vector_query = pd.np.array(tfidfer.transform([' '.join(tokens)]).todense())[0]
    topic_vector_query = pcaer.transform([tfidf_vector_query])
    query_series = pd.Series(topic_vector_query, index=corpus.columns)
    return corpus_text[query_series.dot(corpus.T).values.argmax()]


"""
tfidf_search('Hello world, do you have a cat?')

topic_search('Hello world, do you have a cat?')
# 'Do you have a pet?'

search('The quick brown fox jumped over the lazy dog')
# 'The dog sat on the floor.'

search('A dog barked at my car incessantly.')
# 'A dog chased the car, barking.'
tokenize('A dog barked at my car incessantly.')
# ['dog', 'bark', 'at', 'car', 'incessantli']

search('A Rotweiller barked at my car incessantly.')
# 'The cat hated getting in the car.'
tokenize('A Rotweiller barked at my car incessantly.')
# ['rotweil', 'bark', 'at', 'car', 'incessantli']

list(df.columns)
# ['ate', 'can', 'car', 'cat', 'chase', 'cute', 'die', 'dog', 'ferret', 'flower', 'hair', 'hat', 'have', 'it', 'kitten', 'pet', 'ran',
#   'squirrel', 'struck', 'took', 'tree', 'trick', 'turtl', 'up', 'vet', 'water'],
