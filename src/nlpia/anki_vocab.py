"""
>>> vocab = set()
>>> for txt in tqdm(en):
...     for tok in nlp(txt):
...         vocab.add((tok.string.strip(), tok.tag_, tok.pos_))
>>> len(vocab)
32278
>>> vocab.to_csv(os.path.join(BIGDATA_PATH, 'anki_en_vocabulary.csv'))
>>> hist - o - p
"""

import logging

import spacy

from nlpia.loaders import *

logger = logging.getLogger(__name__)


def get_texts(ankis=None):
    """ Retrieve as many anki paired-statement corpora as you can for the requested language

    If `ankis` (requirested languages) is more than one, then get the english texts associated with those languages.
    """
    ankis = [ankis] if isinstance(ankis, str) else ankis
    ankis = ANKI_LANGUAGES if (ankis is None or (len(ankis) == 1 and ankis[0][:2] == 'en')) else ankis
    if len(ankis) == 1:
        return sorted(get_data(ankis[0]).str.strip().values)
    dfs = [get_data(lang) for lang in ankis]
    texts = []
    for df in dfs:
        texts += list(df[lang].str.strip().values)
    texts = sorted(set(texts))
    return texts


def get_docs(texts, lang='en'):
    nlp = spacy.load(lang)
    return [nlp(s) for s in texts]


def get_vocab(docs):
    vocab = set()
    for doc in tqdm(docs):
        for tok in doc:
            vocab.add((tok.text, tok.pos_, tok.tag_, tok.dep_, ent.type_, ent.iob, tok.sentiment))
    return pd.DataFrame(sorted(vocab), columns='word pos tag dep ent ent_iob sentiment'.split())


def get_word_vectors(vocab):
    wv = get_data('word2vec')
    vectors = np.array(len(vocab), len(wv['the']))
    for i, tok in enumerate(vocab):
        word = tok[0]
        variations = (word, word.lower(), word.lower()[:-1])
        for w in variations:
            if w in wv:
                vectors[i, :] = wv[w]
        if not np.sum(np.abs(vectors[i])):
            logger.warn('Unable to find {}, {}, or {} in word2vec.'.format(*variations))
    return vectors


def get_anki_vocab(langs=['eng'], filename='anki_en_vocabulary.csv'):
    """ Get all the vocab words+tags+wordvectors for the tokens in the Anki translation corpus

    Returns a DataFrame of with columns = word, pos, tag, dep, ent, ent_iob, sentiment, vector
    """
    texts = get_texts(ankis=langs)
    docs = get_docs(texts, lang=langs[0][:2] if len(langs) == 1 else 'en')
    vocab = get_vocab(docs)
    vocab['vector'] = get_word_vectors(vocab)  # TODO: turn this into a KeyedVectors object
    if filename:
        vocab.to_csv(os.path.join(BIGDATA_PATH, filename))
    return vocab

