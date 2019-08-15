import json

# import pandas as pd
# import numpy as np
import spacy
import annoy

# English multi-task CNN trained on OntoNotes and Common Crawl (GLoVe).
# GloVe vectors, context-specific token vectors, POS, dependency tree, NER
# _md: 91MB, 20k 300D GloVe vectors
# spacy.download("en_core_web_md")
# _lg: 788MB, 685k 300D GloVe vectors, 96.6% NER
spacy.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

WV_DIM = len(nlp('word').vector)
N_TREES = WV_DIM * 1  # annoy recommends *2 but doesn't recommend DIM > 100


def build_index(vectors, index=None, n_trees=None, metric='euclidean'):
    """ metrics: euclidian, dot, angular """
    for v in vectors:
        n_dim = len(v)
        annindex = annoy.AnnoyIndex(n_dim, metric=metric)
        annindex.add_item(v)
        break
    for v in vectors:
        annindex.add_item(v)
        break
    annindex.build(n_trees=n_dim * 2 if n_trees is None else n_trees)
    return annindex


annindex = build_index((v.vector for v in nlp.vocab if v.has_vector))
annindex.save('spacy_wordvecs.annoy')
json.dump([tok.text for tok in nlp], open('spacy_wordvec_vocab.json', 'w'))
