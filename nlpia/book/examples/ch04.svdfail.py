import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(svd_topic_vectors, sms.spam, test_size=0.5, random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['svd16_spam'] = lda.predict(pca_topic_vectors)
from nlpia.data.loaders import get_data
sms = get_data('sms-spam')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import casual_tokenize
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_cov = tfidf_docs.dot(tfidf_docs.T)

from sklearn.decomposition import TruncatedSVD
from seaborn import plt
svd = TruncatedSVD(16)
svd = svd.fit(tfidf_cov)
svd_topic_vectors = svd.transform(tfidf_cov)

import pandas as pd
svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
                                 columns=['topic{}'.format(i) for i in range(16)])

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(svd_topic_vectors, sms.spam, test_size=0.5, random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['svd16_spam'] = lda.predict(svd_topic_vectors)
round(float(lda.score(X_test, y_test)), 3)
hist - o - p
hist
svd_topic_vectors
# %paste
# >>> svd = TruncatedSVD(16)  # <1>
# >>> svd = svd.fit(tfidf_cov)
# >>> svd_topic_vectors = svd.transform(tfidf_cov)
# >>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
#                                      columns=['topic{}'.format(i) for i in range(16)])

svd_topic_vectors.head()
tfidf_cov
pd.DataFrame(tfidf_cov, columns=['doc{}'.format(i) for i in range(len(tfidf_cov))])
columns = ['doc{}'.format(i) for i in range(len(tfidf_cov))]
pd.DataFrame(tfidf_cov, columns=columns, index=index)
pd.DataFrame(tfidf_cov, columns=columns, index=columns)
pd.DataFrame(tfidf_cov, columns=columns, index=columns).head()
# hist -o -p
# >>> headings = ['doc{}'.format(i) for i in range(len(tfidf_cov))]
# >>> pd.DataFrame(tfidf_cov, columns=headings, index=headings).round(3).head()
# >>> svd = TruncatedSVD(16)  # <1>
# >>> svd = svd.fit(tfidf_cov)
# >>> svd_topic_vectors = svd.transform(tfidf_cov)
# >>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
#                                      columns=['topic{}'.format(i) for i in range(16)])
# >>> svd_topic_vectors.head()
# >>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
#                                      columns=['topic{}'.format(i) for i in range(16)],
#                                      index=['doc{}'.format(i) for i in range(len(tfidf_cov))])
# >>> svd_topic_vectors.head()
# >>> svd_topic_vectors = svd.transform(tfidf_cov)
# >>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
#                                      columns=['topic{}'.format(i) for i in range(16)],
#                                      index=['doc{}'.format(i) for i in range(len(tfidf_cov))])
# >>> svd_topic_vectors.head()
# svd.transform?
# >>> svd_topic_vectors = svd.fit_transform(tfidf_docs)
# >>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
#                                      columns=['topic{}'.format(i) for i in range(16)],
#                                      index=['doc{}'.format(i) for i in range(len(tfidf_cov))])
# >>> svd_topic_vectors.round(3).head()

sms.spam[:5]
svd_topic_vectors.iloc[:3].corr()
svd_topic_vectors.iloc[:6].dot(svd_topic_vectors.iloc[:6].T).round(3)
sms.spam[:6]
svd_topic_vectors = svd.transform(tfidf_cov)
n = len(sms)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors,
                                 columns=['topic{}'.format(i) for i in range(16)],
                                 index=['sms{}{}'.format(i, '!' * j) for (i, j) in zip(range(n), sms.spam)])
svd_topic_vectors.round(3).head()
svd_topic_vectors.round(3).head(6)
hist - o - p
svd_topic_vectors.iloc[:6].dot(svd_topic_vectors.iloc[:6].T).round(3)
from numpy import linalg
ans = linalg.svd(tfidf_cov)
ans
U, S, V = linalg.svd(tfidf_docs)
U
U.shape
len(sms)
tfidf_docs.shape
utopic_vectors = U.dot(tfidf_docs)
U.shape
S.diag()
S
S.round(3)
V.shape
tfidf_docs.shape
V[:16, :].dot(tfidf_docs.T)
vtopic_vectors = V[:16, :].dot(tfidf_docs.T)
v_topic_vectors.iloc[:6].dot(v_topic_vectors.iloc[:6].T).round(3)
v_topic_vectors = V[:16, :].dot(tfidf_docs.T)
v_topic_vectors.iloc[:6].dot(v_topic_vectors.iloc[:6].T).round(3)
v_topic_vectors[:6, :].dot(v_topic_vectors[:6, :].T).round(3)
sim = v_topic_vectors[:6, :].dot(v_topic_vectors[:6, :].T)
sim /= sim.diag()
sim /= np.diag(sim)
sim /= pd.np.diag(sim)
sim
sim.round(3)
sim.round(6)
sim.round(9)
(sim * 1e10).round(6)
(sim * 1e15).round(6)
sim
V[:16, :].dot(tfidf_docs.T).round(3)
V[:16, :].dot(tfidf_docs.T).round(2)
V[:16, :].dot(tfidf_docs.T).round(2)[:6, :6]
V
V.shape
S.shape
U.shape
Vt = V[:, :16]
Ut = U[:16, :]
Vt.dot(tfidf_docs.T).round(2)[:6, :6]
Vt.dot(tfidf_docs).round(2)[:6, :6]
Vt.T.dot(tfidf_docs.T).round(2)[:6, :6]
utopics = Ut.T.dot(tfidf_docs.T)
utopics = Ut.dot(tfidf_docs.T)
utopics = Ut.dot(tfidf_docs)
utopics
utopics.round(3)
utopics.round(3).shape
utopics.dot(utopics.T).T[:6]
utopics.dot(utopics.T).T[:6].round(3)
utopics.dot(utopics.T).shape
utopics.T.dot(utopics).shape
utopics.T.dot(utopics)[:6].round(3)
utopics.T.dot(utopics)[:6].round(3).shape
utopics
utopics.shape
utopics.T.dot(utopics).shape
utopics.T.dot(utopics)[:6, :6].round(3)
utopics.T.dot(utopics)[:6, :6].round(3) * 100
C = utopics.T.dot(utopics)[:6, :6]
C
C.round(3)
C2 = C / diag(C)
C2 = C / pd.np.diag(C)
C2
C2.round(3)
pd.DataFrame(C2).round(3)
