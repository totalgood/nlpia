# Chapter 3 examples

from collections import Counter

import pandas as pd
from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction import TfidfVectorizer

CORPUS = ['"Hello world!"', 'Go fly a kite.', 'Kite World', 'Take a flying leap!', 'Should I fly home?']


def tfidf_corpus(docs=CORPUS):
    """ Count the words in a corpus and return a TfidfVectorizer() as well as all the TFIDF vecgtors for the corpus

    Args:
      docs (iterable of strs): a sequence of documents (strings)

    Returns:
      (TfidfVectorizer, tfidf_vectors)
    """
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(docs)
    return vectorizer, vectorizer.transform(docs)


def BM25Score(query_str, vectorizer, tfidfs, k1=1.5, b=0.75):
    query_tfidf = vectorizer.transform([query_str])[0]
    scores = []

    for idx, doc in enumerate(self.DocTF) :
        commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
        tmp_score = []
        doc_terms_len = self.DocLen[idx]
        for term in commonTerms :
            upper = (doc[term] * (k1+1))
            below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
            tmp_score.append(self.DocIDF[term] * upper / below)
        scores.append(sum(tmp_score))
    return scores