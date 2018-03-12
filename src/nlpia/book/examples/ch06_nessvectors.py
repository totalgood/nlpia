""" From chapter 6 about the "ness" of word vectors:

Compose a word vector for each "ness" like placeness, humanness, femaleness, etc
Provide a nessvector for any given word

>>> nessvector('Seattle')
placeness      0.257971
peopleness    -0.059435
animalness    -0.014691
conceptness    0.175459
femaleness    -0.359303
dtype: float64
>>> nessvector('Portland')
placeness      0.365310
peopleness    -0.198677
animalness     0.065087
conceptness    0.020675
femaleness    -0.252396
dtype: float64
>>> nessvector('Marie_Curie')
placeness     -0.463387
peopleness     0.354787
animalness     0.171099
conceptness   -0.320268
femaleness     0.257770
dtype: float64
>>> nessvector('Timbers')
placeness     -0.039665
peopleness     0.279271
animalness    -0.328952
conceptness    0.187153
femaleness    -0.097807
dtype: float64

>>> nessvector('Marie_Curie').round(2)
placeness     -0.46
peopleness     0.35
animalness     0.17
conceptness   -0.32
femaleness     0.26
dtype: float64


TODO:
automate the search for synonyms with higher than 60% similarity, walking a shallow graph
"""

import pandas as pd
from nlpia.data.loaders import get_data
from gensim.models.keyedvectors import KeyedVectors

wordvector_path = get_data('word2vec')
word_vectors = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)


###################################################
# Still need to create a class derived from gensim's Word2vec model instead of relying on word_vectors global


COMPONENT_WORDS = [
    ('placeness', ('geography Geography geographic geographical geographical_location location ' +
                   'locale locations proximity').split()),
    ('peopleness', 'human Humans homo_sapiens peole people individuals humankind people men women'.split()),
    ('animalness', 'animal mammal carnivore animals Animal animal_welfare dog pet cats ani_mal'.split()),
    ('conceptness', 'concept concepts idea'.split()),
    ('femaleness', 'female Female females femal woman girl lady'.split()),
]


def component_vector(words=COMPONENT_WORDS[0][1]):
    vector = pd.np.zeros(300)
    for word in words:
        v = word_vectors[word]
        vector += v / len(words)
    return vector


COMPONENTS = pd.DataFrame([component_vector(words) for (component, words) in COMPONENT_WORDS],
                          index=[component for (component, words) in COMPONENT_WORDS])


def nessvector(target, components=COMPONENTS):
    target = word_vectors[target] if isinstance(target, str) else target
    vector = word_vectors.cosine_similarities(target, components.values)
    return pd.Series((vector - vector.mean()) / .15, index=components.index)


#
##############################################################


word_vectors['Marie_Curie']

word_vectors['place'].std()
word_vectors['place'].min()
word_vectors['place'].max()


word_vectors.most_similar('place')
word_vectors.most_similar('location')
word_vectors.most_similar('geography')

placeness = pd.np.zeros(300)
for word in ('geography Geography geographic geographical geographical_location location' +
             'locale locations proximity').split():
    v = word_vectors[word]
    print(v.min(), v.max())
    placeness += v
placeness /= 9.

word_vectors.cosine_similarities(placeness,
                                 [word_vectors[word] for word in
                                  'place geography location address position'.split()])

word_vectors.most_similar('animal')


animalness = pd.np.zeros(300)

animalness = pd.np.zeros(300)
for word in 'animal mammal carnivore animals Animal animal_welfare dog pet cats ani_mal'.split():
    v = word_vectors[word]
    print(v.min(), v.max())
    animalness += v / 10.
word_vectors.similar_by_vector(animalness)


word_vectors.most_similar('people')
word_vectors.most_similar('humans')

peopleness = pd.np.zeros(300)
for word in 'human Humans homo_sapiens peole people individuals humankind people men women'.split():
    v = word_vectors[word]
    print(v.min(), v.max())
    peopleness += v / 10.

word_vectors.similar_by_vector(peopleness)
word_vectors.similar_by_vector(animalness)
word_vectors.similar_by_vector(placeness)


target = word_vectors['Marie_Curie']
word_vectors.cosine_similarities(target, [peopleness, animalness, placeness])


word_vectors.most_similar('concept')

conceptness = pd.np.zeros(300)
for word in 'concept concepts idea'.split():
    v = word_vectors[word]
    print(v.min(), v.max())
    conceptness += v / 3.


target = word_vectors['Marie_Curie']
word_vectors.cosine_similarities(target, [peopleness, animalness, placeness, conceptness])


word_vectors.most_similar('female')
word_vectors.most_similar('woman')

femaleness = pd.np.zeros(300)
for word in 'female Female females femal woman girl lady'.split():
    v = word_vectors[word]
    femaleness += v / 7.

word_vectors.similar_by_vector(conceptness)
word_vectors.similar_by_vector(femaleness)


target = word_vectors['Marie_Curie']
mc_nessvector = word_vectors.cosine_similarities(target, [peopleness, animalness, placeness, conceptness, femaleness])

