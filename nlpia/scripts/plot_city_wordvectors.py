import numpy as np
np.__file__
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from seaborn import plt
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
capitals = pd.read_html('https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States')[0]
import pickle
pickle.load(open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'rb'))
tsne = pickle.load(open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'rb'))
tsne.embedding_plus
len(tsne.embedding)
len(tsne.embedding_)
embeddings = tsne.embedding_plus; embeddings.merge(capitals, how='inner', on=['name', 'country'])
capitals.columns
capitals = pd.read_html('https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States', header=0)[0]
capitals.columns
capitals.columns = s.replace(' ', '_').replace('-', '_').lower()
capitals.columns = [s.replace(' ', '_').replace('-', '_').lower() for s in capitals.columns]
columns = capitals.columns
list(enumerate(columns))
columns[5] = 'area_sqmi'
columns = list(capitals.columns)
columns[5] = 'area_sqmi'
columns[6] = 'population_2010'
columns[8] = 'unnamed_8'
columns[9] = 'unnamed_9'
columns[1] = 'abbreviation'
capitals.columns = columns
capitals.to_csv('/data/nlpia/us_capitals.csv.gz', compression='gzip')
capitals2 = pd.read_csv('/data/nlpia/us_capitals.csv.gz')
capitals2.index
capitals.index
capitals.State
capitals.state
capitals.abbreviation
capitals2 = pd.read_csv('/data/nlpia/cities.csv.gz')
cities = pd.read_csv('/data/nlpia/cities.csv.gz', low_memory=False)
cities.columns
cities[cities.country_code == 'US'].head()
columns
list(enumerate(columns))
columns[1] = 'admin1_code'
columns[3] = 'city'
capitals.columns = columns
columns = list(cities.columns)
list(enumerate(columns))
columns[1] = 'city'
capitals['country_code'] = 'US'
cities2 = cities.merge(capitals, how='inner', on=['city', 'admin1_code', 'country_code'])
cities.city
columns
cities.columns = columns
cities2 = cities.merge(capitals, how='inner', on=['city', 'admin1_code', 'country_code'])
cities2
cities2 = cities.merge(capitals, how='left', on=['city', 'admin1_code', 'country_code'])
len(cities2)
len(cities)
cities3 = cities[['city', 'admin1_code', 'country_code']]
us = cities3[cities3.country_code = 'us']
us = cities3[cities3.country_code == 'us']
us.city
us = cities3[cities3.country_code == 'US']
us.city
[s for s in capitals.city if s not in us.city.values]
len(capitals.city)
len(us.city)
cities2 = cities.merge(capitals, how='inner', on=['city', 'admin1_code', 'country_code'])
[s for s in capitals.city if s not in cities2.city.values]
[s for s in capitals.city if s not in cities.city.values]
for s in capitals:
    print(s)
    break
cities[cities.city == 'Hartford']
cities['hartford' in s.lower().strip() for s in cities.name]
cities[['hartford' in s.lower().strip() for s in cities.name]]
capitals
capitals.x
capitals
capitals.columns[-2:]
capitals.loc[capitals.columns[-2:][0]]
capitals[capitals.columns[-2:][0]]
capitals[capitals.columns[-3:][0]]
capitals.columns[-3:][0]
capitals.columns[-4:][0]
capitals[capitals.columns[-4:][0]]
capitals[capitals.columns[0]]
capitals[capitals.columns[1]]
capitals[capitals.columns[2]]
capitals[capitals.columns[1]]
del capitals.loc[0]
capitals.loc[0].drop()
capitals = capitals.loc[1:]
capitals
capitals[capitals.columns[1]]
capitals[capitals.columns[2]]
capitals[capitals.columns[3]]
capitals[capitals.columns[4]]
capitals['capital_since'] = capitals.capital_since.astype('int')
capitals['capital_since'] = capitals.loc[:,'capitals_since'].astype('int')
capitals['capital_since'] = capitals.loc[:,'capital_since'].astype('int')
capitals.loc[:,'capital_since'] = capitals.capital_since.astype('int')
capitals.loc[:,'capital_since'] = capitals.loc[:,'capital_since'].astype('int')
capitals.loc[:,'capital_since'] = capitals.loc[:,'capital_since'].values.astype('int')
capitals['capital_since'] = capitals.capital_since.values.astype('int')
capitals.loc[:,'capital_since'] = list(capitals.capital_since.values.astype('int'))
capitals.loc[range(len(capitals)),'capital_since'] = list(capitals.capital_since.values.astype('int'))
capitals['capital_since'] = list(capitals.capital_since.values.astype('int'))
del capitals['capital_since']
capitals2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States', header=0)[0]
capitals2 = capitals2.loc[1:].copy()
columns = list(capitals2.columns)
list(enumerate(columns))
columns = [s.replace(' ', '_').replace('-', '_').lower() for s in capitals.columns]
columns[8] = 'unnamed_8'
columns[9] = 'unnamed_9'
columns[5] = 'area_sqmi'
columns[1] = 'admin1_code'
list(enumerate(columns))
columns = list(capitals2.columns)
columns = [s.replace(' ', '_').replace('-', '_').lower() for s in columns]
columns[8] = 'unnamed_8'
columns[9] = 'unnamed_9'
list(enumerate(columns))
columns[6] = 'population_2010'
columns[5] = 'area_sqmi'
columns[1] = 'state_abbreviation'
columns
capitals = capitals2
columns[2] = 'statehood_year'
capitals
capitals.columns = columns
capitals
capitals['city'] = capitals.capital
capitals['population'] = capitals.population.astype(int)
capitals['population'] = capitals.population_2010.astype(int)
capitals['capital_since'] = capitals.capital_since.astype(int)
capitals['population_2010_metro'] = capitals.notes.astype(int)
capitals['population_2010_metro'] = capitals.notes.astype(float)
capitals['population_2010_rank'] = capitals.unnamed_8.astype(int)
''.join(re.findall('\w', '\thello_w orld& '))
import re
''.join(re.findall('\w', '\thello_w orld& '))
def remove_invalid_chars(str_or_seq, valid_regex=r'\w'):
    seq = [str_or_seq] if isinstance(str_or_seq, str) else str_or_seq
    seq = [''.join(re.findall(valid_regex, s)) for s in str_or_seq]
    return seq if isinstance(str_or_seq, str) else str_or_seq[0]


def clean_columns(columns, valid_regex=r'\w', lower=True):
    columns = [re.sub('\s', '_', c).lower() for c in columns]
    columns = remove_invalid_chars(columns, valid_chars=r'\w')
    return columns
capitals
capitals.columns = clean_columns(capitals.columns)
%paste
capitals.columns = clean_columns(capitals.columns)
%paste
capitals.columns = clean_columns(capitals.columns)
capitals.columns
capitals.unnamed_8
capitals.unnamed_8 == capitals.population_2010_rank
(capitals.unnamed_8 == capitals.population_2010_rank).all()
del capitals['unnamed_8']
capitals.population_2010_rank
!pwd
!whoami
!uname
hist
cities
cities.columns
cities = cities.sort_values('modification_date')
cities.drop_duplicates(['city', 'admin1_code', 'country_code'], keep=last)
cities.drop_duplicates(['city', 'admin1_code', 'country_code'], keep='last')
cities2 = _
cities.shape
us = cities[(cities.country == 'US') && ~(cities.admin1_code.isnul())]
us = cities[(cities.country == 'US') & ~(cities.admin1_code.isnul())]
us = cities[(cities.country_code == 'US') & ~(cities.admin1_code.isnul())]
us = cities[(cities.country_code == 'US') & ~(cities.admin1_code.isnull())]
us
us.describe()
us['population'] = us.population.astype(int)
us['population'] = us.population.astype('int')
us = us.copy()
us['population'] = us.population.astype('int')
us.head()
us.tail()
us = us[us.timezone.str.startswith('America')].copy()
len(us)
us = cities[(cities.country_code == 'US') & ~(cities.admin1_code.isnull())]
us = us.copy()
us['population'] = us.population.astype('int')
    DATA_PATH = '/data/'; w2v = None;

    w2v = os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin') if w2v is None else w2v
    try:
        model = Word2Vec.loadWord2Vec.load_word2vec_format(w2v, binary=True) if isinstance(w2v, str) else w2v
    except IOError:
        model = os.path.join(DATA_PATH, w2v)
        model = Word2Vec.loadWord2Vec.load_word2vec_format(model, binary=True)
import os
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
    DATA_PATH = '/data/'; w2v = None;

    w2v = os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin') if w2v is None else w2v
    try:
        model = Word2Vec.loadWord2Vec.load_word2vec_format(w2v, binary=True) if isinstance(w2v, str) else w2v
    except IOError:
        model = os.path.join(DATA_PATH, w2v)
        model = Word2Vec.loadWord2Vec.load_word2vec_format(model, binary=True)
    DATA_PATH = '/data/'; w2v = None;

    w2v = os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin') if w2v is None else w2v
    try:
        model = Word2Vec.load_word2vec_format(w2v, binary=True) if isinstance(w2v, str) else w2v
    except IOError:
        model = os.path.join(DATA_PATH, w2v)
        model = Word2Vec.loadWord2Vec.load_word2vec_format(model, binary=True)
vocab = [s.replace(' ', '_') for s in us.city]
capitals
capitals.columns
us.columns
del us['modification_date']
us['state'] = us.admin1_code
us['state'].apply(dict(zip(capitals.state_abbreviation, capitals.state)))
us['state'].map(dict(zip(capitals.state_abbreviation, capitals.state)))
us['state'] = us['state'].map(dict(zip(capitals.state_abbreviation, capitals.state)))
us['state_abbreviation'] = us.admin1_code
vocab = [s.replace(' ', '_').strip() for s in us.city] + [s.replace(' ', '_').strip() for s in us.state] + [s.replace(' ', '_').strip() for s in us.state_abbreviation]
vocab = [s.replace(' ', '_').strip() for s in us.city] + [s.replace(' ', '_').strip() for s in us.state] + [s.replace(' ', '_').strip() for s in us.state_abbreviation]
us.state_abbreviation.value_counts()
us.state.value_counts()
us.city.value_counts()
us.city.dtype
vocab = [s.replace(' ', '_').strip() for s in us.city] + [s.replace(' ', '_').strip() for s in us.state] + [s.replace(' ', '_').strip() for s in us.state_abbreviation]
vocab = [s.replace(' ', '_').strip() for s in us.city]
vocab += [s.replace(' ', '_').strip() for s in us.state]
vocab += [s.replace(' ', '_').strip() for s in us.state if isinstance(s, str)]
vocab += [s.replace(' ', '_').strip() for s in us.state_abbreviation]
vocab = set(vocab)
vocab = set([s for s in vocab if s in model.wv])
vocab
'MS' in vocab
'AL' in vocab
'AK' in vocab
'Alaska' in vocab
'Pascagoula' in vocab
'Portland' in vocab
v300 = pd.DataFrame([model.wv[s] for s in vocab], index=list(vocab), columns=range(300))
us300 = pd.DataFrame([model.wv[a] + model.wv[b] + model.wv[c] for a, b, c in zip(us.city, us.state, us.state_abbreviation) if a.replace(' ', '_').strip() in vocab], index=us.index)
us300 = pd.DataFrame([model.wv[a] + model.wv[b] + model.wv[c] for a, b, c in zip(us.city, us.state, us.state_abbreviation) if a in vocab], index=us.index)
us.city_ = us.city.str.replace(' ', '_')
us.state_ = us.state.str.replace(' ', '_').strip()
us['state_'] = us.state.str.replace(' ', '_').str.strip()
us['city_'] = us.city.str.replace(' ', '_').str.strip()
us300 = pd.DataFrame([model.wv[a] + model.wv[b] + model.wv[c] for a, b, c in zip(us.city_, us.state_, us.state_abbreviation) if a in vocab], index=us.index)
'South_Carolina' in vocab
'South Carolina' in vocab
'South Carolina' in model.wv
us300 = pd.DataFrame([model.wv[a] + (model.wv[b] if b in vocab else zeros(300)) + model.wv[c] for a, b, c in zip(us.city_, us.state_, us.state_abbreviation) if a in vocab], index=us.index)
us300 = pd.DataFrame([model.wv[a] + (model.wv[b] if b in vocab else pd.np.zeros(300)) + model.wv[c] for a, b, c in zip(us.city_, us.state_, us.state_abbreviation) if a in vocab], index=us.index)
us300 = pd.DataFrame([[i] + list(model.wv[a] + (model.wv[b] if b in vocab else 0) + model.wv[c]) for a, b, c, i in zip(us.city_, us.state_, us.state_abbreviation, us.index) if a in vocab])
us300.set_index(0, drop=True)
us300 = us300.set_index(0, drop=True)
us300.describe()
us300.T.describe()
us300.describe()
norms = pd.np.array([pd.np.linalg.norm(row) for i, row in us300.iterrows()])
norms.max()
norms.min()
norms = pd.np.array([pd.np.linalg.norm(row) for row in model.wv.values()])
dir(models.wv)
dir(model.wv)
model.wv.vocab
model.wv.index2word
model.wv.index2word.values
list(model.wv.index2word)
norms = pd.np.array([pd.np.linalg.norm(model.wv[s]) for s in list(model.wv.index2word)])
norms
norms.mean()
norms = pd.np.array([pd.np.sum(model.wv[s].abs()) for s in list(model.wv.index2word)])
norms = pd.np.array([pd.np.sum(pd.np.abs(model.wv[s])) for s in list(model.wv.index2word)])
norms.max()
norms.min()
norms = pd.np.array([pd.np.sum(model.wv[s]) for s in list(model.wv.index2word)])
norms.max()
norms.min()
norms = pd.np.array([pd.np.linalg.norm(model.wv[s]) for s in list(model.wv.index2word)])
norms.max()
norms.min()
model.most_similar(positive=['Portland', 'OR', 'Oregon'])
model.similar_by_vector(model.wv['Portland'] + model.wv['OR'] + model.wv['Oregon'])
v = model.wv['Portland'] + model.wv['OR'] + model.wv['Oregon']
v /= pd.np.linalg.norm(v)
model.similar_by_vector(v)
v1, v2, v3 = model.wv['Portland'], model.wv['OR'], model.wv['Oregon']
v = (v1 + v2) / pd.np.linalg.norm(v1 + v2) + v3
v /= pd.np.linalg.norm(v)
model.similar_by_vector(v)
from pugnlp import utils
v1.dot(v2) / pd.np.linalg.norm(v1) / pd.np.linalg.norm(v2)
v1.dot(v3) / pd.np.linalg.norm(v1) / pd.np.linalg.norm(v3)
v1.dot(v3)
(v1 - v3)
pd.np.linalg.norm(v1 - v3)
v1.dot(v3)
v1.dot(v3) / (pd.np.linalg.norm(v1) * pd.np.linalg.norm(v3))
model.similarity('Portland', 'Oregon')
model['Portland']
(model['Portland'] == model.wv['Portland']).all()
us
us300
[s for s in set(us.state) if s.replace(' ', '_') if s not in model.wv]
[s for s in set(us.state) if str(s).replace(' ', '_').strip() if s not in model.wv]
[s for s in set(us.state) if str(s).replace(' ', '_').strip() not in model.wv]
us300 = pd.DataFrame([[i] + list(model.wv[a] + (model.wv[b] if b in vocab else model.wv[c]) + model.wv[c]) for a, b, c, i in zip(us.city_, us.state_, us.state_abbreviation, us.index) if a in vocab])
us300 = us300.set_index(0, drop=True)
us300
tsne = TSNE?
from sklearn.decomposition import PCA
pca = PCA()
pca = PCA(n_components=2)
pca.fit(us300)
us2pca = pca.transform(us300)
us2pca = pd.DataFrame(us2pca, columns=list('xy'))
us2pca = pd.DataFrame(us2pca, columns=list('xy'), index=us300.index)
us2pca = pd.DataFrame(us2pca, columns=list('xy'), index=[', '.join(s) for s in zip(us.city[us300.index], us.state[us300.index])])
us300.index
us2pca = pd.DataFrame(us2pca, columns=list('xy'), index=[', '.join([str(c) for c in s]) for s in zip(us.city[us300.index], us.state_abbreviation[us300.index])])
us2pca
index=[', '.join([c for c in s]) for s in zip(us.city[us300.index], us.state_abbreviation[us300.index])]
index=[', '.join(s) for s in zip(us.city[us300.index], us.state_abbreviation[us300.index])]
index = [', '.join(s) for s in zip(us.city[us300.index], us.state_abbreviation[us300.index])]
us300.index = index
pca = PCA(n_components=2)
uspca2 = pca.fit_transform(us300)
uspca2
len(uspca2)
len(us300)
len(us300.index)
len(index)
index
uspca2 = pd.DataFrame(uspca2, columns=list('xy'), index=index)
uspca2
ax = uspca2.plot.scatter('x', 'y', figsize=(6,9))
def annotate(row, ax):
    """Add a text label to the plot of a DataFrame indicated by the provided axis (ax).
    
    Reference:
       https://stackoverflow.com/a/40979683/623735
    """
    print(row['name'])
    ax.annotate(row['name'], (row.x, row.y), xytext=(7, -5), textcoords='offset points')
    return row['name']
uspca2['name'] = uspca2.index.values
usbig.apply(annotate, ax=ax, axis=1)
uspca2.sample(100).apply(annotate, ax=ax, axis=1)
plt.show()
from seaborn import plt
import seaborn
from matplotlib import plt
from matplotlib import pyplot as plt
plt.show()
us300.index
len(us)
us[us300.index]
us['full_name'] = [', '.join(s) for s in zip(us.city, us.state_abbreviation)]
mask = [fn for fn in us.full_name if fn in us300.index.values]
mask = np.array(mask)
mask.sum()
len(mask)
len(us300)
mask = np.array([fn in us300.index.values for fn in us.full_name])
us300meta = us[mask]
us300meta.set_index('full_name', drop=True)
(uspca2.index == us300meta.index).all()
(uspca2.index == us300.index).all()
us300meta = us300meta.reindex()
(uspca2.index == us300meta.index).all()
(uspca2.index == us300meta.index).sum()
us300meta = us300meta[uspca2.index]
us300meta = us300meta.loc[uspca2.index]
us300meta = us300meta.loc[uspca2.index.values]
us300meta = us300meta[uspca2.index.values]
us300meta.index
us300meta.set_index('full_name', drop=True)
us300meta = us300meta.set_index('full_name', drop=False)
us300meta = us300meta[uspca2.index.values]
us300meta
us300meta.index
us300meta.columns
us300meta.geonameid
us300meta = us300meta[uspca2.index]
rows300meta = [us300meta.loc[i] for i in uspca2.index.values]
uspca2.index.values
rows = pd.DataFrame(rows, index=uspca2.index, columns=us300meta.columns)
rows = pd.DataFrame(rows300meta)
rows = pd.DataFrame(rows300meta, index=uspca2.index, columns=us300meta.columns)
len(set(uspca2.index.values))
len(uspca2.index)
len(set(us300.index.values))
us300.columns
len(set(uspca2.index.values))
us300['name'] = us300.index.values
uspca2['name'] = uspca2.index.values
us300.drop_duplicates('name', keep='last')
us300meta['name'] = us300meta.index.values
us300meta.drop_duplicates('name', keep='last')
uspca2.drop_duplicates('name', keep='last')
len(uspc2)
len(uspca2)
us300meta = us300meta.drop_duplicates('name', keep='last')
uspca2 = uspca2.drop_duplicates('name', keep='last')
us300 = us300.drop_duplicates('name', keep='last')
us300
len(uspca2)
len(us300meta)
len(us300)
(uspca2.index == us300met.index).all()
(uspca2.index == us300meta.index).all()
(uspca2.index == us300.index).all()
uspca2['population'] = us300meta.population
us300 = us300.drop_duplicates('name', keep='last').copy()
uspca2 = uspca2.drop_duplicates('name', keep='last').copy()
us300meta = us300meta.drop_duplicates('name', keep='last').copy()
uspca2['population'] = us300meta['population'].copy()
(uspca2.index == us300.index).all()
(uspca2.index == us300meta.index).all()
uspca2big = uspca2[uspca2.population > 150000].copy()
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
uspca2big = uspca2[uspca2.population > 210000].copy()
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
count, mask = Counter(), []
for name in uspca2.name:
    state = name[-2:]
    count += Counter([state])
    if count[state] < 5:
        mask += [True]
    else:
        mask += [False]
from collections import Counter
count, mask = Counter(), []
for name in uspca2.name:
    state = name[-2:]
    count += Counter([state])
    if count[state] < 5:
        mask += [True]
    else:
        mask += [False]
count, mask = Counter(), []
for name in uspca2.name:
    state = name[-2:]
    count += Counter([state])
    if count[state] < 5:
        mask += [True]
    else:
        mask += [False]
uspca2 = uspca2.sort_values('population', ascending=False)
uspca2
count, mask = Counter(), []
for name in uspca2.name:
    state = name[-2:]
    count += Counter([state])
    if count[state] < 5:
        mask += [True]
    else:
        mask += [False]
uspca2big = uspca2[mask]
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
uspca2big = uspca2[mask & (uspca2.population > 230000)]
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
uspca2big.x = -uspca2big.x
uspca2big['x'] = uspca2big['x']
uspca2big['x'] = uspca2big['x'].copy()
uspca2big['x'] = uspca2big['x'].values
uspca2big['x'] = uspca2big['x'].values.copy()
uspca2big = uspca2[mask & (uspca2.population > 230000)].copy()
uspca2big.x = -uspca2big.x
uspca2big.y = -uspca2big.y
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
count, mask = Counter(), []
for name in uspca2.name:
    state = name[-2:]
    count += Counter([state])
    if count[state] < 3:
        mask += [True]
    else:
        mask += [False]
uspca2big = uspca2[mask & (uspca2.population > 250000)].copy()
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
ax = uspca2big.plot.scatter('x', 'y', figsize=(10,7))
uspca2big.apply(annotate, ax=ax, axis=1)
plt.show()
us
us.to_csv('src/nlpia/nlpia/data/cities_us.csv.gz', compression='gzip')
us300.to_csv('src/nlpia/nlpia/data/cities_us_wordvectors.csv.gz', compression='gzip')
uspca2.to_csv('src/nlpia/nlpia/data/cities_us_wordvectors_pca2.csv.gz', compression='gzip')
uspca2big.to_csv('src/nlpia/nlpia/data/cities_us_wordvectors_plot.csv.gz', compression='gzip')
uspca10 = pca.fit_transform(us300)
uspca10 = pca.fit_transform(us300[us300.columns[:300]])
us300[us300.columns[:300]].shape
uspca10.to_csv('src/nlpia/nlpia/data/cities_us_wordvectors_pca10.csv.gz', compression='gzip')
uspca10 = pd.DataFrame(uspca10, columns=list('xy'))
uspca10.index
uspca10.index=us300.index
uspca10.to_csv('src/nlpia/nlpia/data/cities_us_wordvectors_pca10.csv.gz', compression='gzip')
len(uspca2)
tsne = TSNE()
tsne = tsne.fit(uspca10)
embeddings = pd.DataFrame(tsne.embedding_, index=uspca10.index, columns=list('xy'))
len(us)
len(us300meta)
us300meta.columns
for c in us300meta.columns:
    if c not in embeddings.columns:
        embeddings[c] = us300meta[c]
tsne.embeddings_plus = embeddings
(uspca10.index == us300meta.index).all()
pickle.dump(tsne, open('/home/hobs/src/nlpia/nlpia/data/tsne_us_cities_w2v.pickle2', 'wb'), protocol=2, fix_imports=True)
pickle.dump(tsne, gzip.open('/home/hobs/src/nlpia/nlpia/data/tsne_us_cities_w2v.pickle2.gz', 'wb'), protocol=2, fix_imports=True)
import gzip
pickle.dump(tsne, gzip.open('/home/hobs/src/nlpia/nlpia/data/tsne_us_cities_w2v.pickle2.gz', 'wb'), protocol=2, fix_imports=True)
