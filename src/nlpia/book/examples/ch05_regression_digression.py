# A Regression Digression

##########################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from nlpia.data.loaders import get_data

sms = get_data('sms-spam')
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs, columns=list(zip(*sorted([(v, k) for (k, v) in tfidf.vocabulary_.items()])))[1])


##########################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# TFIDF->LDA
tfidf_lda = LDA(n_components=1)
tfidf_lda.fit(tfidf_docs, sms.spam)
# Most spammy terms (words) in the TFIDF:
tfidf_lda_coef = pd.DataFrame(list(zip(tfidf_lda.coef_[0, :], tfidf_docs.columns)), columns='coef term'.split())
print(tfidf_lda_coef.sort_values('coef', ascending=False).head())
#               coef                 term
# 2666  7.606693e+06              darling
# 7168  5.393772e+06             sexychat
# 895   5.202198e+06                80488
# 6085  4.865422e+06              parties
# 9025  4.852177e+06  www.07781482378.com


##########################

from sklearn.decomposition import PCA

# TFIDF->PCA->LDA
pca = PCA(n_components=256)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=['topic{}'.format(i) for i in range(pca.n_components_)])
pca_components = pd.DataFrame(pca.components_,
                              columns=tfidf_docs.columns,
                              index=['topic{}'.format(i) for i in range(pca.n_components_)])


##########################

pca_lda = LDA(n_components=1)
pca_lda.fit(pca_topic_vectors, sms.spam)
sms['pca_lda_spam_prob'] = pca_lda.predict_proba(pca_topic_vectors)[:, 1]


##########################

pca_topic_vectors['spam_label_'] = sms.spam
print(pca_topic_vectors.corr().spam_label_.sort_values(ascending=False).head())
# spam_label_    1.000000
# topic4         0.564850
# topic2         0.275897
# topic10        0.186002
# topic9         0.167077


##########################

sms['topic4'] = pca_topic_vectors.topic4

##########################

from nltk.sentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
scores = pd.DataFrame([vader.polarity_scores(text) for text in sms.text])
sms['vader'] = scores['compound']
sms.describe().tail()
#      spam  pca_lda_spam_prob   vader
# min   0.0       5.845537e-15 -0.9042
# 25%   0.0       2.155922e-09  0.0000
# 50%   0.0       2.822494e-08  0.0000
# 75%   0.0       1.172246e-06  0.5267
# max   1.0       1.000000e+00  1.0000


##########################

mask = (sms.vader > 0.1) | (sms.vader < -0.1)
sms = sms[mask].copy()


##########################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
for col in ['pca_lda_spam_prob', 'vader', 'topic4']:
    sms.loc[:, col] = scaler.fit_transform(sms[[col]])


##########################

from nlpia.models import LinearRegressor

line = LinearRegressor()
line = line.fit(sms['topic4'], sms['vader'])
print('{:.4f}'.format(line.slope))
# 0.29

sms['line'] = line.predict(sms['topic4'])


##########################

from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(n_iter=20000)
sgd = sgd.fit(sms[['topic4']], sms['vader'])
print('{:.4f}'.format(sgd.coef_[0]))
# 0.2930

sms['sgd'] = sgd.predict(sms[['topic4']])


##########################

from nlpia.models import OneNeuronRegressor

nn = OneNeuronRegressor(alpha=100, n_iter=200)
nn = nn.fit(sms[['topic4']], sms['vader'])
print(nn.W[0, 1])
# 0.29386408

sms['neuron'] = nn.predict(sms[['topic4']])


##########################

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn


def sentiment_scatter(sms=sms):
    sms = sms.sort_values('topic4').sample(200)
    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(1, 1, 1)
    ax = sms.plot.scatter(x='topic4', y='line', ax=ax, color='g',   marker='+', s=400, lw=3, alpha=.5)
    ax = sms.plot.scatter(x='topic4', y='sgd', ax=ax, color='r',    marker='x', s=200, lw=3, alpha=.5)
    ax = sms.plot.scatter(x='topic4', y='vader', ax=ax, color='k',  marker='s', s=100, alpha=.5)
    ax = sms.plot.scatter(x='topic4', y='neuron', ax=ax, color='c', marker='.', s=100, alpha=.5)
    ax = sms.plot.scatter(x='topic4', y='pca_lda_spam_prob', ax=ax, marker='o', s=150, color='b', alpha=.5)
    plt.ylabel('Sentiment')
    plt.xlabel('Topic 4')
    plt.legend(['LinearRegressor', 'SGDRegressor', 'VADER', 'OneNeuronRegresor', 'PCA->LDA->spaminess'])
    plt.tight_layout()
    plt.grid()
    plt.show()


sentiment_scatter()
