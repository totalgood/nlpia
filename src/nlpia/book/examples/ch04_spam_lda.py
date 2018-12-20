import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120  # <1>

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]  # <2>
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
mask = sms.spam.astype(bool).values
sms['spam'] = sms.spam.astype(int)
"""
>>> sms.head(6)
       spam                                               text
sms0      0  Go until jurong point, crazy.. Available only ...
sms1      0                      Ok lar... Joking wif u oni...
sms2!     1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3      0  U dun say so early hor... U c already then say...
sms4      0  Nah I don't think he goes to usf, he lives aro...
sms5!     1  FreeMsg Hey there darling it's been 3 week's n...
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
"""

mask = sms.spam.astype(bool).values  # <1>
spam_centroid = tfidf_docs[mask].mean(axis=0) # <2>
ham_centroid = tfidf_docs[~mask].mean(axis=0)
"""
>>> mask = sms.spam.astype(bool)
>>> spam_centroid = tfidf_docs[mask].mean(axis=0)
>>> spam_centroid.round(2)
array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
>>> ham_centroid = tfidf_docs[~mask].mean(axis=0)
>>> ham_centroid.round(2)
array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])
"""

spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
"""
>>> spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
>>> spamminess_score
array([-0.01469806, -0.02007376,  0.03856095, ..., -0.01014774, -0.00344281,  0.00395752])
"""

from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)

"""
>>> from sklearn.preprocessing import MinMaxScaler
>>> sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
>>> sms['lda_predict'] = (sms.lda_score > .5).astype(int)
>>> sms['spam lda_predict lda_score'.split()].round(2).head(6)
       spam  lda_predict  lda_score
sms0      0            0       0.23
sms1      0            0       0.18
sms2!     1            1       0.72
sms3      0            0       0.18
sms4      0            0       0.29
sms5!     1            1       0.55
"""

"""
>>> from sklearn.decomposition import PCA
>>> from matplotlib import pyplot as plt
>>> import seaborn
>>> pca_model = PCA(n_components=3)
>>> tfidf_docs_3d = pca_model.fit_transform(tfidf_docs)
>>> df = pd.DataFrame(tfidf_docs_3d)
>>> ax = df[~mask].plot(x=0, y=1, kind='scatter', alpha=.5, c='green')
>>> df[mask].plot(x=0, y=1, ax=ax, alpha=.1, kind='scatter', c='red')
>>> plt.xlabel(' x')
>>> plt.ylabel(' y')
>>> plt.savefig('spam_lda_2d_scatter.png')
"""
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn
pca_model = PCA(n_components=3)
tfidf_docs_3d = pca_model.fit_transform(tfidf_docs)
df = pd.DataFrame(tfidf_docs_3d)
ax = df[~mask].plot(x=0, y=1, kind='scatter', alpha=.5, c='green')
df[mask].plot(x=0, y=1, ax=ax, alpha=.1, kind='scatter', c='red')
plt.xlabel(' x')
plt.ylabel(' y')
plt.savefig('spam_lda_2d_scatter.png')


import plotly as py
spam_trace = dict(
        x=df[0][mask], y=df[1][mask], z=df[2][mask],
        type="scatter3d", mode='markers',
        marker= dict(size=3, color='red', line=dict(width=0)) 
    )
ham_trace = dict(
        x=df[0][~mask], y=df[1][~mask], z=df[2][~mask],
        type="scatter3d", mode='markers',
        marker= dict(size=3, color='green', line=dict(width=0)) 
    )
fig = dict(data=[ham_trace, spam_trace], layout={'title': 'LDA Spamminess Model'})
py.offline.plot(fig, filename='lda_spam_3d_scatter.html')


"""
>>> sms['spam lda_predict lda'.split()].round(2)
      spam  lda_predict   lda
0        0            0  0.23
1        0            0  0.18
2        1            1  0.72
3        0            0  0.18
4        0            0  0.29
5        1            1  0.55
6        0            0  0.32
7        0            0  0.50
8        1            1  0.89
9        1            1  0.77
10       0            0  0.24
11       1            1  0.79
12       1            1  0.92
13       0            0  0.38
14       0            1  0.55
15       1            1  0.53
16       0            0  0.13
17       0            0  0.25
18       0            0  0.28
19       1            1  0.63
20       0            0  0.27
21       0            0  0.40
22       0            0  0.16
23       0            0  0.22
24       0            0  0.25
25       0            0  0.34
26       0            0  0.39
27       0            0  0.22
28       0            0  0.34
29       0            0  0.28
...    ...          ...   ...
4807     0            0  0.44
4808     0            0  0.30
4809     0            0  0.33
4810     0            0  0.32
4811     0            0  0.38
4812     1            1  0.75
4813     0            0  0.26
4814     0            0  0.37
4815     0            0  0.21
4816     0            0  0.31
4817     0            0  0.44
4818     0            0  0.36
4819     0            0  0.39
4820     0            0  0.39
4821     0            0  0.24
4822     0            0  0.34
4823     0            0  0.39
4824     0            0  0.26
4825     0            0  0.33
4826     0            0  0.41
4827     0            0  0.12
4828     0            0  0.33
4829     0            0  0.37
4830     0            0  0.25
4831     1            1  0.69
4832     1            1  0.85
4833     0            0  0.29
4834     0            0  0.27
4835     0            0  0.33
4836     0            0  0.40

[4837 rows x 3 columns]
>>> (sms.lda_predict - sms.spam).abs()
0       0
1       0
2       0
3       0
4       0
5       0
6       0
7       0
8       0
9       0
10      0
11      0
12      0
13      0
14      1
15      0
16      0
17      0
18      0
19      0
20      0
21      0
22      0
23      0
24      0
25      0
26      0
27      0
28      0
29      0
       ..
4807    0
4808    0
4809    0
4810    0
4811    0
4812    0
4813    0
4814    0
4815    0
4816    0
4817    0
4818    0
4819    0
4820    0
4821    0
4822    0
4823    0
4824    0
4825    0
4826    0
4827    0
4828    0
4829    0
4830    0
4831    0
4832    0
4833    0
4834    0
4835    0
4836    0
Length: 4837, dtype: int64
>>> (sms.lda_predict - sms.spam).abs().sum() / len(sms)
0.022534628902212115
>>> (sms.lda_predict - sms.spam.astype).abs()[sms.spam.astype(bool)].sum() / sms.spam.sum()
>>> (sms.lda_predict - sms.spam.astype)[sms.spam.astype(bool)].abs().sum() / sms.spam.sum()
>>> (sms.lda_predict - sms.spam.astype)[sms.spam.astype(bool)]
>>> (sms.lda_predict - sms.spam)[sms.spam.astype(bool)]
2       0
5       0
8       0
9       0
11      0
12      0
15      0
19      0
34      0
42      0
54     -1
56      0
65      0
67      0
68     -1
93      0
95      0
114     0
117     0
120     0
121     0
123     0
134     0
135     0
139     0
147     0
159     0
160     0
164     0
165     0
       ..
4607    0
4629    0
4630    0
4631    0
4633    0
4635   -1
4642    0
4643    0
4646    0
4692    0
4708    0
4714    0
4721    0
4725    0
4727    0
4731    0
4732    0
4733    0
4747    0
4752    0
4757    0
4762    0
4766    0
4789    0
4791    0
4802    0
4805    0
4812    0
4831    0
4832    0
Length: 638, dtype: int64
>>> (sms.lda_predict - sms.spam)[sms.spam.astype(bool)].abs().sum() / sms.spam.sum()
0.07053291536050156
"""