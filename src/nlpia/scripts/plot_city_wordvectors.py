from gensim.models import KeyedVectors
import matplotlib
matplotlib.use('TkAgg')  # noqa

import pandas as pd
import numpy as np

from nlpia.data.loaders import get_data


wv = None


def plot_city_wordvectors():
    global wv
    path = get_data('wv')
    wv = KeyedVectors.load_word2vec_format(path, binary=True) if wv is None else wv
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


    vocab = pd.np.concatenate([us.city, us.st, us.state])
    vocab = np.array([word for word in vocab if word in wv.wv])
    vocab[:10]
    # array(['Edna', 'Henderson', 'Natalia', 'Yorktown', 'Brighton', 'Berry',
    #        'Trinity', 'Villas', 'Bessemer', 'Aurora'], dtype='<U15')


    # >>> us_300D = pd.DataFrame([[i] + list(wv[c] + (wv[state] if state in vocab else wv[s]) + wv[s]) for i, c, state, s
    # ...                         in zip(us.index, us.city, us.state, us.st) if c in vocab])
    city_plus_state = []
    for c, state, st in zip(us.city, us.state, us.st):
        if c not in vocab:
            continue
        row = []
        if state in vocab:
            row.extend(wv[c] + wv[state])
        else:
            row.extend(wv[c] + wv[st])
        city_plus_state.append(row)
    us_300D = pd.DataFrame(city_plus_state)


    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)  # <1>
    us_300D = get_data('cities_us_wordvectors')
    us_2D = pca.fit_transform(us_300D.iloc[:, :300])  # <2>
    # ----
    # <1> PCA here is for visualization of high dimensional vectors only, not for calculating Word2vec vectors
    # <2> The last column (# 301) of this DataFrame contains the name, which is also stored in the DataFrame index.
