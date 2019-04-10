# Manually load w2v
# import os
# from nlpia.data.loaders import BIGDATA_PATH
# from gensim.models import KeyedVectors
# path = os.path.join(BIGDATA_PATH, 'GoogleNews-vectors-negative300.bin.gz')
# wv = KeyedVectors.load_word2vec_format(path, binary=True)

# nlpia can now automatically download and load w2v

from nlpia.data.loaders import get_data
from gensim.models import KeyedVectors
wv = get_data('word2vec')
# wv = KeyedVectors.load_word2vec_format(path, binary=True)
len(wv.vocab)
# 3000000
wv.vectors.shape
# (3000000, 300)


import pandas as pd
vocab = pd.Series(wv.vocab)
vocab.iloc[100000:100006]  # different words for new KeyedVector format
# Illington_Fund             Vocab(count:447860, index:2552140)
# Illingworth                 Vocab(count:2905166, index:94834)
# Illingworth_Halifax       Vocab(count:1984281, index:1015719)
# Illini                      Vocab(count:2984391, index:15609)
# IlliniBoard.com           Vocab(count:1481047, index:1518953)
# Illini_Bluffs              Vocab(count:2636947, index:363053)


import numpy as np
np.linalg.norm(wv['Illinois'] - wv['Illini'])  # <1>
# 3.3653798
similarity = np.dot(wv['Illinois'], wv['Illini']) / (
    np.linalg.norm(wv['Illinois']) * np.linalg.norm(wv['Illini']))   # <2>
similarity
# 0.5501352
1 - similarity  # <3>
# 0.4498648
# ----
# <1> Euclidean distance
# <2> Cosine similarity
# <3> Cosine distance


wv['Illini']
# array([ 0.15625   ,  0.18652344,  0.33203125,  0.55859375,  0.03637695,
#        -0.09375   , -0.05029297,  0.16796875, -0.0625    ,  0.09912109,
#        -0.0291748 ,  0.39257812,  0.05395508,  0.35351562, -0.02270508,


from nlpia.data.loaders import get_data
cities = get_data('cities')
cities.head(1).T
# geonameid                       3039154
# name                          El Tarter
# asciiname                     El Tarter
# alternatenames     Ehl Tarter,Эл Тартер
# latitude                        42.5795
# longitude                       1.65362
# feature_class                         P
# feature_code                        PPL
# country_code                         AD
# cc2                                 NaN
# admin1_code                          02
# admin2_code                         NaN
# admin3_code                         NaN
# admin4_code                         NaN
# population                         1052
# elevation                           NaN
# dem                                1721
# timezone                 Europe/Andorra
# modification_date            2012-11-03


us = cities[(cities.country_code == 'US') & (cities.admin1_code.notnull())].copy()
states = pd.read_csv('http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv')
states = dict(zip(states.Abbreviation, states.State))
us['city'] = us.name.copy()
us['st'] = us.admin1_code.copy()
us['state'] = us.st.map(states)
us[us.columns[-3:]].head()
#                      city  st    state
# geonameid
# 4046255       Bay Minette  AL  Alabama
# 4046274              Edna  TX    Texas
# 4046319    Bayou La Batre  AL  Alabama
# 4046332         Henderson  TX    Texas
# 4046430           Natalia  TX    Texas


import numpy as np
vocab = pd.np.concatenate([us.city, us.st, us.state])
vocab = np.array([word for word in vocab if word in wv.wv])
vocab[:5]
# array(['Edna', 'Henderson', 'Natalia', 'Yorktown', 'Brighton', 'Berry',
#        'Trinity', 'Villas', 'Bessemer', 'Aurora'], dtype='<U15')
vocab[:10]
# array(['Edna', 'Henderson', 'Natalia', 'Yorktown', 'Brighton', 'Berry',
#        'Trinity', 'Villas', 'Bessemer', 'Aurora'], dtype='<U15')


# >>> us_300D = pd.DataFrame([[i] + list(wv[c] + (wv[state] if state in vocab else wv[s]) + wv[s]) for i, c, state, s
# ...                         in zip(us.index, us.city, us.state, us.st) if c in vocab])
city_plus_state = []
us = us.sort_values('population', ascending=False)
for c, state, st in zip(us.city, us.state, us.st):
    if c not in vocab:
        continue
    row = []
    if state in vocab:
        row.extend(wv[c] + wv[state])
    else:
        row.extend(wv[c] + wv[st])
    city_plus_state.append(row)
us_300D_sorted = pd.DataFrame(city_plus_state)
del city_plus_state
del wv


# Simplified plot of 10 largest cities
from sklearn.decomposition import PCA
pca = PCA(n_components=2) 
us_300D = get_data('cities_us_wordvectors')
us_2D = pca.fit_transform(us_300D.iloc[:10, :300])

# Original confusing/complicated/detailed plot
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # <1>
us_300D = get_data('cities_us_wordvectors')
us_2D = pca.fit_transform(us_300D.iloc[:500, :300])  # <2>
# ----
# <1> PCA here is for visualization of high dimensional vectors only, not for calculating Word2vec vectors
# <2> The last column (# 301) of this DataFrame contains the name, which is also stored in the DataFrame index.

# this example is completely independent of the examples above
from nlpia.data.loaders import get_data
from nlpia.plots import offline_plotly_scatter_bubble
df = get_data('cities_us_wordvectors_pca2_meta')
df = df.sort_values('population', ascending=False)[:10].copy()
df[['x', 'y']] = - df[['x', 'y']]  # <1> flip East/West & North/South axes to match geography better
html = offline_plotly_scatter_bubble(
    df, x='x', y='y',
    size_col=None, text_col='name', category_col=None,
    xscale=None, yscale=None,  # 'log' or None
    layout={}, marker={'sizeref': 3000})
with open('wordmap.html', 'w') as fout:
    fout.write(html)
# !firefox ./wordmap.html
""" Simplified Plot

from nlpia.data.loaders import get_data
from nlpia.plots import offline_plotly_scatter_bubble
df = get_data('cities_us_wordvectors_pca2_meta')
df = df.sort_values('population', ascending=False)[:10].copy()
df[['x', 'y']] = - df[['x', 'y']]  # <1>
html = offline_plotly_scatter_bubble(
    df, x='x', y='y',
    size_col=None, text_col='name', category_col=None,
    xscale=None, yscale=None,  # 'log' or None
    layout={}, marker={'sizeref': 3000})
with open('wordmap.html', 'w') as fout:
    fout.write(html)
# !firefox ./wordmap.html

<1> flips the East/West North/South axes around to match geography better


"""
