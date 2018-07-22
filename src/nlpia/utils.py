# -*- coding: utf-8 -*-
import logging
import sys
import os

import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

from nlpia.constants import DATA_PATH
from nlpia.data.loaders import get_data


def stdout_logging(loglevel=logging.INFO):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(lineno)d: %(message)s"

    logging.config.dictConfig(level=loglevel, stream=sys.stdout,
                              format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def embed_wordvecs(w2v=None, df=None, vocab='name', embedder=TSNE, **kwargs):
    w2v = os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin') if w2v is None else w2v
    try:
        model = KeyedVectors.load_word2vec_format(w2v, binary=True) if isinstance(w2v, str) else w2v
    except IOError:
        model = os.path.join(DATA_PATH, w2v)
        model = KeyedVectors.loadWord2Vec.load_word2vec_format(model, binary=True)
    if df is None:
        df = get_data('cities')
    if isinstance(vocab, str) and vocab in df.columns:
        vocab = set([s.replace(' ', '_') for s in vocab.name] + [s.replace(' ', '_') for s in df.country])

    vocab = [word for word in vocab if word in model.wv]
    vectors = pd.DataFrame([model.wv[word] for word in vocab], index=vocab, columns=range(300))
    tsne = embedder(**kwargs)
    tsne = tsne.fit(vectors)
    return pd.DataFrame(tsne.embedding_, columns=['x', 'y'])


"""
Trial and error to produce the steps encoded in embed_vectors() above

```python
'xy'.split()
embeddings = pd.DataFrame(tsne.embedding_, columns=list('xy'))
embeddings
embeddings.plot(kind='scatter', x='x', y='y')
plt.show()
embeddings = pd.DataFrame(tsne.embedding_, columns=list('xy'), index=vocab)
embeddings
!pip install seaborn
from seborn import plt
from seaborn import plt
from seaborn import plt
plt = seaborn.plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn.plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn
plt = seaborn.plt
from matplotlib import plt
from seaborn import *
embeddings.plot(kind='scatter', x='x', y='y')
row.x
embeddings.apply??
embeddings.apply(print, axis=1)
embeddings.apply(annotate, ax=ax, axis=1)
plt.show()
pd.read_csv('/data/cities.csv.gz', low_memory=False, header=0, index_col=0)
cities = _
vocab = set([s.replace(' ', '_') for s in cities.name] + [s.replace(' ', '_') for s in cities.country])
cities.columns
cities.alternatenames
cities.columns
cities.dem
cities.country_code
cp /data/AdWords\ API\ Location\ Criteria\ 2017-06-26.csv /data/nlpia/
cp /data/cities.csv.gz /data/nlpia/
cities.columns
cities.cc2.dropna()
cities.country_code.dropna()
cities.admin1_code
cities.admin2_code
cities.admin2_code.dropna()
cities.admin3_code.dropna()
vocab = set([s.replace(' ', '_') for s in cities.name])
vocab = [s.replace(' ', '_') for s in cities.name]
vocab = set(vocab)
set
del set
set
vocab = set(vocab)
vocab
vocab = [word for word in vocab if word in model.wv]
vocab

df
us = [s.replace(' ', '_') for s in cities.name[(cities.country_code == 'US') & (cities.population > 100000)] if s in model.wv]
len(us)
embeddings = pd.DataFrame(tsne.embedding_, columns=list('xy'), index=vectors.index)
us = [s.replace(' ', '_') for s in cities.name[(cities.country_code == 'US') & (cities.population > 100000)] if s.replace(' ', '_') in model.wv]
len(us)
cities['name_'] = [s.replace(' ', '_') for s in cities.name]
import pickle
import gzip
pickle.dump(tsne, gzip.open('/data/nlpia/tsne_world_cities_w2v.pickle.gz', 'wb'))
len(tsne.embedding_)
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle', 'wt'))
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle', 'w'))
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle', 'wb'))
pickle.dump?
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'wb'), protocol=2, fix_imports=True)
embeddings
len(cities)
embeddings['latitude'] = [cities.latitude[s.replace('_', ' ')] for s in embeddings.index.values]
cities.latitude.index
embeddings['latitude'] = [cities.latitude[cities.name == s.replace('_', ' ')] for s in embeddings.index.values]
embeddings[['longitude', 'altitude']] = [cities[['longitude', 'altitude']][cities.name == s.replace('_', ' ')] for s in embeddings.index.values]
cities.columns
embeddings[['longitude', 'elevation']] = [cities[['longitude', 'elevation']][cities.name == s.replace('_', ' ')] for s in embeddings.index.values]
embeddings.index
embeddings.merge(cities, how='outer', right='name_')
embeddings.merge(cities, how='outer', right_on='name_')
embeddings.merge(cities, how='outer', left_on=embeddings.index, right_on='name_')
embeddings.merge(cities, how='left', left_index=True, right_on='name_')
embeddings2 = _
len(embeddings)
len(embeddings2)
embeddings.merge(cities, how='left', left_on=embeddings.index.values, right_on='name_')
embeddings.merge(cities, how='right', left_on=embeddings.index.values, right_on='name_')
embeddings.merge(cities, how='inner', left_on=embeddings.index.values, right_on='name_')
embeddings['name_'] = embeddings.index.values
embeddings
embeddings.merge(cities, how='inner', on='name_')
embeddings = embeddings.merge(cities, how='inner', on='name_')
embeddings.isnull().sum()
tsne.embeddings_plus = embeddings
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'wb'), protocol=2, fix_imports=True)
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle', 'wb'))
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle3', 'wb'))
pickle.dump(tsne, gzip.open('/data/nlpia/tsne_world_cities_w2v.pickle2.gz', 'wb'), protocol=2, fix_imports=True)
ax = us.plot(kind='scatter', x='x', y='y')
us = embeddings[embeddings.country_code=='US']
len(us)
us.drop_duplicates()
us.info
us.describe()
us.dtype
us.dtypes
us.latitude_x
us.latitude
us.longitude
embeddings = tsne.embeddings_
embeddings = tsne.embedding_
embeddings = pd.DataFrame(tsne.embedding_, columns=list('xy'))
len(vectors)
len(embeddings)
embeddings.index = vectors.index
embeddings.reindex()
embeddings = embeddings.reindex()
embeddings['name_'] = embeddings.index.values
embeddings = embeddings.merge(cities, how='inner', on='name_')
embeddings[embeddings.country_code == 'US'].plot(kind='scatter', x='x', y='y')
embeddings
embeddings.apply(annotate, ax=ax, axis=1)
plt.clf()
us = embeddings[embeddings.country_code == 'US']
usbig = us[us.population > 150000]
usbig
'Portland' in embeddings.name_
usbig = us[us.population > 100000]
'Portland' in embeddings.name
embeddings.name
us
set(embeddings.name)
'NYC' in embeddings.name
'NY' in embeddings.name
'New York' in embeddings.name
'Langley' in embeddings.name
u'Langley' in embeddings.name
b'Langley' in embeddings.name
b'Langley' in set(embeddings.name)
'Langley' in embeddings.index
embeddings.loc['Langley']
embeddings.index
embeddings.index.values
embeddings.name_
Richton in embeddings.name_
'Richton' in embeddings.name_
embeddings.name_.str.strip()
embeddings['name_'] = embeddings.name_.str.strip()
'Richton' in embeddings.name_
embedings.loc[42925]
embeddings.loc[42925]
embeddings.loc[42925].name
embeddings.loc[42925].name_
embeddings.loc[42925].name_ == 'Richton'
'Richton' in embeddings.name_
'Richton' in embeddings.loc[42925]
'Richton' in embeddings.loc[42925].values
'Richton' in embeddings.name_.values
'Richton' in embeddings.name.values
'Portland' in embeddings.name_.values
'Portland' in usbig.name_.values
usbig.plot.scatter('x', 'y')
plt.clf()
ax = usbig.plot.scatter('x', 'y')
usbig.apply(annotate, ax=ax, axis=1)
plt.show()
embeddings['name'] = embeddings.name_.str.replace('_', ' ')
embeddings.name
ax = usbig.plot.scatter('x', 'y')
usbig.apply(annotate, ax=ax, axis=1)
plt.show()
plt.close('all')
usbig = embeddings[(embeddings.population >= 200000) & (embeddings.country_code == 'US')]
embeddings.columns
embeddings.drop_duplicates(['name', 'country_code'])
len(tsne.embedding_)
embeddings.drop_duplicates('name')
len(tsne.embedding_)
embeddings = embeddings.drop_duplicates('name', keep='last')
tsne.embedding_plus = embeddings
pickle.dump(tsne, gzip.open('/data/nlpia/tsne_world_cities_w2v.pickle2.gz', 'wb'), protocol=2, fix_imports=True)
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'wb'), protocol=2, fix_imports=True)
usbig = embeddings[(embeddings.population >= 200000) & (embeddings.country_code == 'US')]
ax = usbig.plot.scatter('x', 'y')
usbig.apply(annotate, ax=ax, axis=1)
plt.show()
embeddings.set_index('name', drop=False)
embeddings = embeddings.set_index('name', drop=False)
tsne.embedding_plus = embeddings
del tsne.embeddings_plus
pickle.dump(tsne, gzip.open('/data/nlpia/tsne_world_cities_w2v.pickle2.gz', 'wb'), protocol=2, fix_imports=True)
pickle.dump(tsne, open('/data/nlpia/tsne_world_cities_w2v.pickle2', 'wb'), protocol=2, fix_imports=True)
ax = usbig.plot.scatter('x', 'y')
usbig.apply(annotate, ax=ax, axis=1)
def annotate(row, ax):
    \"\"\"Add a text label to the plot of a DataFrame indicated by the provided axis (ax).

    Reference:
       https://stackoverflow.com/a/40979683/623735
    \"\"\"
    print(row['name'])
    ax.annotate(row['name'], (row.x, row.y), xytext=(7, -5), textcoords='offset points')
    return row['name']
ax = usbig.plot.scatter('x', 'y')
usbig.apply(annotate, ax=ax, axis=1)
plt.show()
```
"""
