import numpy as np

"""
>>> import pandas as pd
>>> from nlpia.data.loaders import get_data
>>> pd.options.display.width = 120  # <1>
>>> sms = get_data('sms-spam')
>>> index = ['sms{}{}'.format(i, '!'*j) for (i,j) in\
...     zip(range(len(sms)), sms.spam)]  # <2>
>>> sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
>>> sms['spam'] = sms.spam.astype(int)
>>> len(sms)
4837
>>> sms.spam.sum()
638
>>> sms.head(6)
      spam                                               text
sms0     0  Go until jurong point, crazy.. Available only ...
sms1     0                      Ok lar... Joking wif u oni...
sms2!    1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3     0  U dun say so early hor... U c already then say...
sms4     0  Nah I don't think he goes to usf, he lives aro...
sms5!    1  FreeMsg Hey there darling it's been 3 week's n...
"""
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120  # <1>
sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in\
    zip(range(len(sms)), sms.spam)]  # <2>
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)
len(sms)

"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf_model.fit_transform(\
...     raw_documents=sms.text).toarray()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(\
    raw_documents=sms.text).toarray()
tfidf_docs.shape
# (4837, 9232)
sms.spam.sum()
# 638

"""
>>> mask = sms.spam.astype(bool)  # <1>
>>> spam_centroid = tfidf_docs[mask].mean(axis=0) # <2>
>>> ham_centroid = tfidf_docs[~mask].mean(axis=0)

>>> spam_centroid.round(2)
array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
>>> ham_centroid.round(2)
array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])
"""
mask = sms.spam.astype(bool)  # <1>
spam_centroid = tfidf_docs[mask].mean(axis=0) # <2>
ham_centroid = tfidf_docs[~mask].mean(axis=0)

spam_centroid.round(2)
# array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
ham_centroid.round(2)
# array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])

"""
>>> spamminess_score = tfidf_docs.dot(spam_centroid -\
...     ham_centroid)  # <1>
>>> spamminess_score.round(2)
array([-0.01, -0.02,  0.04, ..., -0.01, -0.  ,  0.  ])
"""
spamminess_score = tfidf_docs.dot(spam_centroid -\
    ham_centroid)  # <1>
spamminess_score.round(2)

"""
>>> from sklearn.preprocessing import MinMaxScaler
>>> sms['lda_score'] = MinMaxScaler().fit_transform(\
...     spamminess_score.reshape(-1,1))
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
from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(\
    spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)
sms['spam lda_predict lda_score'.split()].round(2).head(6)


"""
>>> (1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3)
0.977
"""
(1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3)
# 0.977

"""
>>> from pugnlp.stats import Confusion
>>> Confusion(sms['spam lda_predict'.split()])
lda_predict     0    1
spam                  
0            4135   64
1              45  593
"""
from pugnlp.stats import Confusion
Confusion(sms['spam lda_predict'.split()])
# lda_predict     0    1
# spam                  
# 0            4135   64
# 1              45  593
# ImportError                               Traceback (most recent call last)
# <ipython-input-3-ab8f002ffa39> in <module>
# ----> 1 from pugnlp.stats import Confusion
#       2 Confusion(sms['spam lda_predict'.split()])

# ~/anaconda3/envs/nlpiaenv/lib/python3.6/site-packages/pugnlp/stats.py in <module>
#      19 from pugnlp.constants import NUMERIC_TYPES
#      20 # watch out for circular import
# ---> 21 from pugnlp.segmentation import stringify
#      22 from pugnlp.util import make_dataframe, listify
#      23 from pugnlp.util import PrettyDict

# ~/anaconda3/envs/nlpiaenv/lib/python3.6/site-packages/pugnlp/segmentation.py in <module>
#      16 import nltk.stem
#      17 
# ---> 18 from pugnlp.detector_morse import Detector
#      19 from pugnlp.detector_morse import slurp
#      20 from pugnlp.futil import find_files

# ~/anaconda3/envs/nlpiaenv/lib/python3.6/site-packages/pugnlp/detector_morse.py in <module>
#      36 from collections import namedtuple
#      37 
# ---> 38 from nlup import case_feature, isnumberlike, listify, BinaryAveragedPerceptron, BinaryConfusion, IO, JSONable
#      39 
#      40 from .penn_treebank_tokenizer import word_tokenize

# ImportError: cannot import name 'IO'

"""
>>> from nlpia.book.examples.ch04_catdog_lsa_3x6x16\
...     import word_topic_vectors
>>> word_topic_vectors.T.round(1)
      cat  dog  apple  lion  nyc  love
top0 -0.6 -0.4    0.5  -0.3  0.4  -0.1
top1 -0.1 -0.3   -0.4  -0.1  0.1   0.8
top2 -0.3  0.8   -0.1  -0.5  0.0   0.1
"""
from nlpia.book.examples.ch04_catdog_lsa_3x6x16\
    import word_topic_vectors
word_topic_vectors.T.round(1)
