""" Functions for accessing the Anki Falshcard app's database of international language flashcards """
import os
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from nlpia.constants import BIGDATA_PATH
from nlpia.loaders import get_data, ANKI_LANGUAGES, LANG2ANKI, nlp

logger = logging.getLogger(__name__)


def get_anki_phrases(lang='english', limit=None):
    """ Retrieve as many anki paired-statement corpora as you can for the requested language

    If `ankis` (requested languages) is more than one, then get the english texts associated with those languages.

    TODO: improve modularity: def function that takes a single language and call it recursively if necessary
    >>> phrases = get_anki_phrases('afr')
    >>> any(('piesang' in s.lower() for s in phrases))
    True

    >> get_anki_phrases('afr')[:2]
    ["'n Groen piesang is nie ryp genoeg om te eet nie.",
     "'n Hond het agter die kat aan gehardloop."]
    """
    lang = lang.strip().lower()[:3]
    lang = LANG2ANKI[lang[:2]] if lang not in ANKI_LANGUAGES else lang
    if lang[:2] == 'en':
        return get_anki_phrases_english(limit=limit)
    return sorted(get_data(lang).iloc[:, -1].str.strip().values)


def get_anki_phrases_english(limit=None):
    """ Return all the English phrases in the Anki translation flashcards

    >>> len(get_anki_phrases_english(limit=100)) > 700
    True
    """
    texts = set()
    for lang in ANKI_LANGUAGES:
        df = get_data(lang)
        phrases = df.eng.str.strip().values
        texts = texts.union(set(phrases))
        if limit and len(texts) >= limit:
            break
    return sorted(texts)


def get_vocab(docs):
    """ Build a DataFrame containing all the words in the docs provided along with their POS tags etc

    >>> doc = nlp("Hey Mr. Tangerine Man!")
    <BLANKLINE>
    ...
    >>> get_vocab([doc])
            word    pos  tag       dep ent_type ent_iob  sentiment
    0          !  PUNCT    .     punct                O        0.0
    1        Hey   INTJ   UH      intj                O        0.0
    2        Man   NOUN   NN      ROOT   PERSON       I        0.0
    3        Mr.  PROPN  NNP  compound                O        0.0
    4  Tangerine  PROPN  NNP  compound   PERSON       B        0.0
    """
    if isinstance(docs, spacy.tokens.doc.Doc):
        return get_vocab([docs])
    vocab = set()
    for doc in tqdm(docs):
        for tok in doc:
            vocab.add((tok.text, tok.pos_, tok.tag_, tok.dep_, tok.ent_type_, tok.ent_iob_, tok.sentiment))
    # TODO: add ent type info and other flags, e.g. like_url, like_email, etc
    return pd.DataFrame(sorted(vocab), columns='word pos tag dep ent_type ent_iob sentiment'.split())


def get_word_vectors(vocab):
    """ Create a word2vec embedding matrix for all the words in the vocab """
    wv = get_data('word2vec')
    vectors = np.array(len(vocab), len(wv['the']))
    for i, tok in enumerate(vocab):
        word = tok[0]
        variations = (word, word.lower(), word.lower()[:-1])
        for w in variations:
            if w in wv:
                vectors[i, :] = wv[w]
        if not np.sum(np.abs(vectors[i])):
            logger.warning('Unable to find {}, {}, or {} in word2vec.'.format(*variations))
    return vectors


def get_anki_vocab(lang=['eng'], limit=None, filename='anki_en_vocabulary.csv'):
    """ Get all the vocab words+tags+wordvectors for the tokens in the Anki translation corpus

    Returns a DataFrame of with columns = word, pos, tag, dep, ent, ent_iob, sentiment, vectors
    """
    texts = get_anki_phrases(lang=lang, limit=limit)
    docs = nlp(texts, lang=lang)
    vocab = get_vocab(docs)
    vocab['vector'] = get_word_vectors(vocab)  # TODO: turn this into a KeyedVectors object
    if filename:
        vocab.to_csv(os.path.join(BIGDATA_PATH, filename))
    return vocab

