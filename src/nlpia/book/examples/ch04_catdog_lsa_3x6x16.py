from nlpia.data.loaders import get_data
from nltk.tokenize import casual_tokenize
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from nlpia.constants import DATA_PATH


NUM_TOPICS = 3
NUM_WORDS = 6
NUM_DOCS = NUM_PRETTY = 16
SAVE_SORTED_CORPUS = ''  # 'cats_and_dogs_sorted.txt'
# import nltk
# nltk.download('wordnet')  # noqa
# from nltk.stem.wordnet import WordNetLemmatizer


# STOPWORDS = 'a an and or the do are with from for of on in by if at to into them'.split()
# STOPWORDS += 'to at it its it\'s that than our you your - -- " \' ? , . !'.split()
STOPWORDS = []

# SYNONYMS = dict(zip(
#     'wolv people person women woman man human he  we  her she him his hers'.split(),
#     'wolf her    her    her   her   her her   her her her her her her her'.split()))
# SYNONYMS.update(dict(zip(
#     'ate pat smarter have had isn\'t hasn\'t no  got get become been was were wa be sat seat sit'.split(),
#     'eat pet smart   has  has not    not     not has has is     is   is  is   is is sit sit  sit'.split())))
# SYNONYMS.update(dict(zip(
#     'i me my mine our ours catbird bird birds birder tortoise turtle turtles turtle\'s don\'t'.split(),
#     'i i  i  i    i   i    bird    bird birds bird   turtle   turtle turtle  turtle    not'.split())))
SYNONYMS = {}

stemmer = None  # PorterStemmer()

pd.options.display.width = 110
pd.options.display.max_columns = 14
pd.options.display.max_colwidth = 32




def normalize_corpus_words(corpus, stemmer=stemmer, synonyms=SYNONYMS, stopwords=STOPWORDS):
    docs = [doc.lower() for doc in corpus]
    docs = [casual_tokenize(doc) for doc in docs]
    docs = [[synonyms.get(w, w) for w in words if w not in stopwords] for words in docs]
    if stemmer:
        docs = [[stemmer.stem(w) for w in words if w not in stopwords] for words in docs]
    docs = [[synonyms.get(w, w) for w in words if w not in stopwords] for words in docs]
    docs = [' '.join(w for w in words if w not in stopwords) for words in docs]
    return docs


def tokenize(text, vocabulary, synonyms=SYNONYMS, stopwords=STOPWORDS):
    doc = normalize_corpus_words([text.lower()], synonyms=synonyms, stopwords=stopwords)[0]
    stems = [w for w in doc.split() if w in vocabulary]
    return stems


fun_words = vocabulary = 'cat dog apple lion nyc love big small'
fun_stems = normalize_corpus_words([fun_words])[0].split()[:NUM_WORDS]
fun_words = fun_words.split()


if SAVE_SORTED_CORPUS:
    tfidfer = TfidfVectorizer(min_df=2, max_df=.6, stop_words=None, token_pattern=r'(?u)\b\w+\b')

    corpus = get_data('cats_and_dogs')[:NUM_DOCS]
    docs = normalize_corpus_words(corpus, stemmer=None)
    tfidf_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    id_words = [(i, w) for (w, i) in tfidfer.vocabulary_.items()]
    tfidf_dense.columns = list(zip(*sorted(id_words)))[1]


    word_tfidf_dense = pd.DataFrame(tfidfer.transform(fun_stems).todense())
    word_tfidf_dense.columns = list(zip(*sorted(id_words)))[1]
    word_tfidf_dense.index = fun_stems
    """
    >>> word_tfidf_dense[fun_stems]
          cat  dog  anim  pet  citi  appl  nyc  car  bike  hat
    cat   1.0  0.0   0.0  0.0   0.0   0.0  0.0  0.0   0.0  0.0
    dog   0.0  1.0   0.0  0.0   0.0   0.0  0.0  0.0   0.0  0.0
    anim  0.0  0.0   1.0  0.0   0.0   0.0  0.0  0.0   0.0  0.0
    pet   0.0  0.0   0.0  1.0   0.0   0.0  0.0  0.0   0.0  0.0
    citi  0.0  0.0   0.0  0.0   1.0   0.0  0.0  0.0   0.0  0.0
    appl  0.0  0.0   0.0  0.0   0.0   1.0  0.0  0.0   0.0  0.0
    nyc   0.0  0.0   0.0  0.0   0.0   0.0  1.0  0.0   0.0  0.0
    car   0.0  0.0   0.0  0.0   0.0   0.0  0.0  1.0   0.0  0.0
    bike  0.0  0.0   0.0  0.0   0.0   0.0  0.0  0.0   1.0  0.0
    hat   0.0  0.0   0.0  0.0   0.0   0.0  0.0  0.0   0.0  1.0
    """

    tfidfer.use_idf = False
    tfidfer.norm = None
    bow_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    bow_dense.columns = list(zip(*sorted(id_words)))[1]
    bow_dense = bow_dense.astype(int)
    tfidfer.use_idf = True
    tfidfer.norm = 'l2'


    """
    >>> tfidf_dense.shape
    (200, 170)
    """


    bow_pretty = bow_dense.copy()
    bow_pretty = bow_pretty[fun_stems]
    bow_pretty['text'] = corpus

    tfidf_pretty = tfidf_dense.copy()
    tfidf_pretty = tfidf_pretty[fun_stems]
    tfidf_pretty['diversity'] = tfidf_pretty[fun_stems].T.sum().values
    tfidf_pretty['text'] = corpus
    # tfidf_pretty['diversity'] = [(row.diversity or 0) / ((float(row.iloc[i % (len(row) - 2)] or 1) ** 2))
    #                              for i, row in tfidf_pretty.iterrows()]
    tfidf_pretty = tfidf_pretty.sort_values('diversity', ascending=False).round(2)
    with open(os.path.join(DATA_PATH, SAVE_SORTED_CORPUS), 'w') as fout:
        fout.write('\n'.join(list(tfidf_pretty.text.values)))

    for col in fun_stems:
        bow_pretty.loc[bow_pretty[col] == 0, col] = ''
    # print(bow_pretty.head())


# do it all over again on a tiny portion of the corpus and vocabulary
corpus = get_data('cats_and_dogs_sorted')[:NUM_PRETTY]
docs = normalize_corpus_words(corpus)
tfidfer = TfidfVectorizer(min_df=1, max_df=.99, stop_words=None, token_pattern=r'(?u)\b\w+\b',
                          vocabulary=fun_stems)
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
bow_pretty = bow_pretty[fun_stems]
bow_pretty['text'] = corpus
for col in fun_stems:
    bow_pretty.loc[bow_pretty[col] == 0, col] = ''
# print(bow_pretty)
word_tfidf_dense = pd.DataFrame(tfidfer.transform(fun_stems).todense())
word_tfidf_dense.columns = list(zip(*sorted(id_words)))[1]
word_tfidf_dense.index = fun_stems

tfidf_pretty = tfidf_dense.copy()
tfidf_pretty = tfidf_pretty[fun_stems]
tfidf_pretty = tfidf_pretty.round(2)
for col in fun_stems:
    tfidf_pretty.loc[tfidf_pretty[col] == 0, col] = ''
# tfidf_pretty[:text]tfidf_pretty.text.str[:16]
# print(tfidf_pretty)


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
# print(tfidf_zeros)
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

pcaer = PCA(n_components=NUM_TOPICS)

doc_topic_vectors = pd.DataFrame(pcaer.fit_transform(tfidf_dense.values), columns=['top{}'.format(i) for i in range(NUM_TOPICS)])
doc_topic_vectors['text'] = corpus
pd.options.display.max_colwidth = 55
# doc_topic_vectors.round(1)
"""
>>> doc_topic_vectors.round(1)
    topic_A  topic_B  topic_C  topic_D                                                             text
0      -0.2      0.1      0.5     -0.2  Animals don't drive cars, but my pet dog likes to stick his ...
1       0.1      0.1     -0.0     -0.3              The Cat in the Hat is not about an animal or a hat.
2       0.1     -0.2      0.5     -0.3                   Cats don't like riding into the city in a car.
3      -0.2     -0.5      0.0      0.3                      Dogs love to chase cars, trucks, and bikes.
4      -0.2     -0.3      0.2      0.1        Wild cats chase bikes and runners but not cars or trucks.
5      -0.1     -0.0      0.6     -0.2              Animals, including pets, don't like riding in cars.
6       0.6      0.0     -0.1      0.0                                 NYC is a city that never sleeps.
7       0.7      0.1      0.0      0.2                                  Come to NYC. See the Big Apple!
8       0.8      0.1      0.0      0.2                                   NYC is known as the Big Apple.
9       0.7      0.1     -0.0      0.1  NYC is the only city where you can hardly find a typical Ame...
10     -0.2     -0.1     -0.5     -0.3                                         It rained cats and dogs.
11     -0.4      0.5     -0.0      0.4                                               I love my pet cat.
12      0.3      0.1     -0.1      0.4                                      I love New York City (NYC).
13     -0.2      0.3      0.2     -0.2                                      He pet the dog on the head.
14     -0.2     -0.5      0.2      0.0                                         Dogs like to chase cars.
15     -0.2     -0.1     -0.3     -0.1                          The cat steered clear of the dog house.
16     -0.1     -0.4      0.3      0.1                                         The car had a bike rack.
"""

word_topic_vectors = pd.DataFrame(pcaer.transform(word_tfidf_dense.values),
                                  columns=['top{}'.format(i) for i in range(NUM_TOPICS)])
word_topic_vectors.index = fun_stems


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


def tfidf_search(text, corpus=tfidf_dense, corpus_text=corpus):
    """ search for the most relevant document """
    tokens = tokenize(text, vocabulary=corpus.columns)
    tfidf_vector_query = np.array(tfidfer.transform([' '.join(tokens)]).todense())[0]
    query_series = pd.Series(tfidf_vector_query, index=corpus.columns)

    return corpus_text[query_series.dot(corpus.T).values.argmax()]


def topic_search(text, corpus=doc_topic_vectors, pcaer=pcaer, corpus_text=corpus):
    """ search for the most relevant document """
    tokens = tokenize(text, vocabulary=corpus.columns)
    tfidf_vector_query = np.array(tfidfer.transform([' '.join(tokens)]).todense())[0]
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
# ['ate', 'can', 'car', 'cat', 'chase', 'cute', 'die', 'dog', 'ferret', 'flower', 'hair',
# 'hat', 'have', 'it', 'kitten', 'pet', 'ran',
#   'squirrel', 'struck', 'took', 'tree', 'trick', 'turtl', 'up', 'vet', 'water'],
"""
U, Sigma, VT = np.linalg.svd(tfidf_dense.T)  # <1> Transpose the doc-word tfidf matrix, because SVD works on column vectors
S = Sigma.copy()
S[4:] = 0
doc_labels = ['doc{}'.format(i) for i in range(len(tfidf_dense))]
U_df = pd.DataFrame(U, index=fun_stems, columns=fun_stems)
VT_df = pd.DataFrame(VT, index=doc_labels, columns=doc_labels)
ndim = 2
truncated_tfidf = U[:, :ndim].dot(np.diag(Sigma)[:ndim, :ndim]).dot(VT.T[:, :ndim].T)
"""
The left singular vectors tell you how to "rotate" the TF-IDF vectors into the topic space, equivalent to creating topics

>>> U_df
       cat   dog  appl   nyc   car  bike   hat
cat  -0.53  0.01 -0.50  0.31 -0.49 -0.00 -0.36
dog  -0.60  0.25  0.19  0.43  0.56 -0.00  0.21
appl -0.16 -0.63  0.17 -0.12  0.37 -0.00 -0.63
nyc  -0.25 -0.69  0.06  0.04 -0.24  0.00  0.63
car  -0.35  0.17  0.32 -0.45 -0.21  0.71 -0.03
bike -0.35  0.17  0.32 -0.45 -0.21 -0.71 -0.03
hat  -0.17  0.00 -0.69 -0.55  0.40  0.00  0.19

>>> VT_df.round(2)
        doc0  doc1  doc2  doc3  doc4  doc5  doc6  doc7  doc8  doc9  doc10  doc11
doc0  -0.37 -0.34 -0.16 -0.22 -0.33 -0.33 -0.27 -0.15 -0.40 -0.40  -0.15  -0.15
doc1   0.19  0.12  0.00  0.01  0.16  0.16 -0.29 -0.51  0.11  0.11  -0.51  -0.51
doc2   0.33  0.11 -0.55 -0.58  0.25  0.25 -0.19  0.11 -0.13 -0.13   0.11   0.11
doc3  -0.31 -0.39 -0.39 -0.27 -0.07 -0.07  0.21 -0.06  0.48  0.48  -0.06  -0.06
doc4   0.03 -0.59  0.28  0.08  0.24  0.24 -0.61  0.14  0.10  0.10   0.14   0.14
doc5   0.00  0.00  0.00 -0.00  0.71 -0.71 -0.00 -0.00  0.00  0.00  -0.00  -0.00
doc6   0.16 -0.51  0.17 -0.10  0.27  0.27  0.62 -0.11 -0.23 -0.23  -0.11  -0.11
doc7  -0.04 -0.07 -0.36  0.41  0.09  0.09  0.00  0.67 -0.06 -0.06  -0.33  -0.33
doc8  -0.54  0.19  0.07 -0.08  0.27  0.27  0.00 -0.00  0.45 -0.55  -0.00  -0.00
doc9  -0.54  0.19  0.07 -0.08  0.27  0.27  0.00 -0.00 -0.55  0.45  -0.00  -0.00
doc10 -0.04 -0.07 -0.36  0.41  0.09  0.09  0.00 -0.33 -0.06 -0.06   0.67  -0.33
doc11 -0.04 -0.07 -0.36  0.41  0.09  0.09  0.00 -0.33 -0.06 -0.06  -0.33   0.67
Try to reconstruct an approximate TFIDF, using only 2 topics (from 7 words):

>>> tfidf_compressed = U[:,:2] @ (pd.np.diag(S)[:2,:] @ VT[:2,:])
>>> tfidf_compressed.shape

array([[ 0.12191697,  0.01013273,  0.04009995, ...,  0.06057937,
         0.07675374, -0.00521042],
       [ 0.0352883 ,  0.07193914,  0.00248612, ...,  0.00928599,
         0.02413519,  0.02511732],
       [ 0.0886943 ,  0.0689894 ,  0.06191171, ...,  0.08041404,
        -0.0090325 ,  0.01368156],
       ...,
       [-0.00116523,  0.00980797,  0.00397578, ..., -0.00540478,
         0.04008382,  0.01717131],
       [ 0.08695312,  0.01880789,  0.04888827, ...,  0.0604496 ,
         0.04363558,  0.00425347],
       [ 0.06353676, -0.01267888,  0.0766847 , ...,  0.07791765,
         0.01870914,  0.00097211]])
"""


if __name__ == '__main__':
    print(word_topic_vectors.T.round(1))