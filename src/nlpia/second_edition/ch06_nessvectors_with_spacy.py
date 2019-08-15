import json

# import pandas as pd
# import numpy as np
import spacy
import annoy

# "en_core_web_lg" and "en_core_web_md"
# English multi-task CNN trained on OntoNotes and Common Crawl (GLoVe).
# GloVe vectors, context-specific token vectors, POS, dependency tree, NER
# _md: 91MB, 20k 300D GloVe vectors
# _lg: 788MB, 685k 300D GloVe vectors, 96.6% NER
# !python -m spacy download "en_core_web_lg"
nlp = spacy.load("en_core_web_lg")

WV_DIM = len(nlp('word').vector)
N_TREES = WV_DIM * 1  # annoy recommends *2 but doesn't recommend DIM > 100


def build_index(vectors, index=None, n_trees=None, metric='euclidean'):
    """ metrics: euclidian, dot, angular """
    vector_iterator = iter(vectors)
    for v in vector_iterator:
        n_dim = len(v)
        annindex = annoy.AnnoyIndex(n_dim, metric=metric)
        annindex.add_item(0, v)
        break
    for i, v in enumerate(vector_iterator):
        annindex.add_item(i + 1, v)
        break
    annindex.build(n_trees=n_dim * 2 if n_trees is None else n_trees)
    return annindex


annindex = build_index((v.vector for v in nlp.vocab if v.has_vector))
annindex.save('spacy_wordvecs.annoy')
json.dump([tok.text for tok in nlp], open('spacy_wordvec_vocab.json', 'w'))
