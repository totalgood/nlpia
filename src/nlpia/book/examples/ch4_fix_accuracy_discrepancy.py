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
"""
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120  # <1>
sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!' * j) for (i, j) in
         zip(range(len(sms)), sms.spam)]  # <2>
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)
len(sms)

"""
>>> import numpy as np
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from nltk.tokenize import casual_tokenize
>>> np.random.seed(42)

>>> counter = CountVectorizer(tokenizer=casual_tokenize)
>>> bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text)\
...     .toarray(), index=index)
>>> column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),\
...     counter.vocabulary_.keys())))
>>> bow_docs.columns = terms
>>> from sklearn.decomposition import LatentDirichletAllocation as LDiA
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
np.random.seed(42)

counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text)
                        .toarray(), index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
                                     counter.vocabulary_.keys())))
bow_docs.columns = terms
from sklearn.decomposition import LatentDirichletAllocation as LDiA


"""
>>> ldia = LDiA(n_components=16, learning_method='batch')
>>> ldia = ldia.fit(bow_docs)  # <1>
>>> ldia.components_.shape
>>> ldia16_topic_vectors = ldia.transform(bow_docs)
>>> ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,\
...     index=index, columns=columns)
>>> ldia16_topic_vectors.round(2).head()
columns = ['topic{}'.format(i) for i in range(16)]
>>> ldia16_topic_vectors = ldia.transform(bow_docs)
>>> ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,\
...     index=index, columns=columns)
>>> ldia16_topic_vectors.round(2).head()
"""
ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)  # <1>
ldia.components_.shape
columns = ['topic{}'.format(i) for i in range(16)]
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,
                                    index=index, columns=columns)
ldia16_topic_vectors.round(2).head()

