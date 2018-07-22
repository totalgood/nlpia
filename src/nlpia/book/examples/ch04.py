"""
[source,python]
----
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> from nlpia.data.loaders import get_data
>>> pd.options.display.width = 120

>>> sms = get_data('sms-spam')
>>> index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]  # <1>
>>> sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
>>> sms.spam = sms.spam.astype(int)
>>> sms.head(6)
      spam                                               text
sms0     0  Go until jurong point, crazy.. Available only ...
sms1     0                      Ok lar... Joking wif u oni...
sms2!    1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3     0  U dun say so early hor... U c already then say...
sms4     0  Nah I don't think he goes to usf, he lives aro...
sms5!    1  FreeMsg Hey there darling it's been 3 week's n...
----
"""

topic = {}
tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), [1, 1, 1, 1, 1, 1])))
topic['pet'] = (.3 * tfidf['cat'] + .3 * tfidf['dog'] + 0 * tfidf['apple']
                + 0 * tfidf['lion'] - .2 * tfidf['NYC'] + .2 * tfidf['love'])
topic['animal'] = (.1 * tfidf['cat'] + .1 * tfidf['dog'] - .1 * tfidf['apple']
                   + .5 * tfidf['lion'] + .1 * tfidf['NYC'] - .1 * tfidf['love'])
topic['city'] = (0 * tfidf['cat'] - .1 * tfidf['dog'] + .2 * tfidf['apple']
                 - .1 * tfidf['lion'] + .5 * tfidf['NYC'] + .1 * tfidf['love'])


word_vector = {}
word_vector['cat'] = .3 * topic['pet'] + .1 * topic['animal'] + 0 * topic['city']
word_vector['dog'] = .3 * topic['pet'] + .1 * topic['animal'] - .1 * topic['city']
word_vector['apple'] = 0 * topic['pet'] - .1 * topic['animal'] + .2 * topic['city']
word_vector['lion'] = 0 * topic['pet'] + .5 * topic['animal'] - .1 * topic['city']
word_vector['NYC'] = -.2 * topic['pet'] + .1 * topic['animal'] + .5 * topic['city']
word_vector['love'] = .2 * topic['pet'] - .1 * topic['animal'] + .1 * topic['city']


import pandas as pd
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')
import matplotlib
matplotlib.use('TkAgg')
import seaborn

from matplotlib import pyplot as plt
from nlpia.data.loaders import get_data

df = get_data('pointcloud').sample(1000)
pca = PCA(n_components=2)
df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y')
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from nlpia.data.loaders import get_data

sms = get_data('sms-spam')
sms.head(3)
#    spam                                               text
# 0     0  Go until jurong point, crazy.. Available only ...
# 1     0                      Ok lar... Joking wif u oni...
# 2     1  Free entry in 2 a wkly comp to win FA Cup fina...


tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs.shape
# (4837, 9232)
sms.spam.sum()
# 638
"""
[source,python]
----
>>> tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
>>> tfidf_docs = pd.DataFrame(tfidf_docs, index=index)
>>> tfidf_docs = tfidf_docs - tfidf_docs.mean()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
----
"""

from sklearn.decomposition import PCA

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=['topic{}'.format(i) for i in range(16)])
pca_topic_vectors.head()
#      topic0    topic1    topic2    topic3    topic4    topic5    topic6 ...
# 0  0.201171  0.002782  0.037215  0.010953 -0.019202 -0.053041  0.039052 ...
# 1  0.404380 -0.093886 -0.077505  0.050910  0.100064  0.047142  0.022761 ...
# 2 -0.030456 -0.048073  0.090164 -0.067071  0.090828 -0.043266 -0.000346 ...
# 3  0.329048 -0.032784 -0.034534 -0.015772  0.052215  0.055835 -0.165424 ...
# 4  0.002159  0.030856  0.038317  0.033872 -0.074714 -0.092599 -0.043725 ...
"""
[source,python]
----
>>> from sklearn.decomposition import PCA

>>> pca = PCA(n_components=16)
>>> pca = pca.fit(tfidf_docs)
>>> pca_topic_vectors = pca.transform(tfidf_docs)
>>> columns = ['topic{}'.format(i) for i in range(pca.n_components)]
>>> pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
>>> pca_topic_vectors.round(3).head()
       topic0  topic1  topic2   ...     topic13  topic14  topic15
sms0    0.201   0.003   0.037   ...      -0.026   -0.019    0.039
sms1    0.404  -0.094  -0.078   ...      -0.036    0.047   -0.036
sms2!  -0.030  -0.048   0.090   ...      -0.017   -0.045    0.057
sms3    0.329  -0.033  -0.035   ...      -0.065    0.022   -0.076
sms4    0.002   0.031   0.038   ...       0.031   -0.081   -0.020
----
"""

# # # can't replicate this or find the svddists code
# import numpy as np
# df = pd.DataFrame(np.array([svddists.reshape(len(sms)), pcadists.reshape(len(sms)), sms.spam])).T,
# # ...              columns='SVD_dist_to_doc3 PCA_dist_to_doc3 spam'.split())
# df.corr()  # < 1 >
# #                   SVD_dist_to_doc3  PCA_dist_to_doc3      spam
# # SVD_dist_to_doc3          1.000000          0.862478 -0.591911
# # PCA_dist_to_doc3          0.862478          1.000000 -0.595148
# # spam                     -0.591911         -0.595148  1.000000

# # <1> `DataFrame.corr()` computes the normalized covariance (similarity or correlation) between all the columns and rows of a DataFrame
