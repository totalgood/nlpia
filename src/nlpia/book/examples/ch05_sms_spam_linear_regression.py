import pandas as pd
from pandas import np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from nltk.tokenize.casual import casual_tokenize
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn  # noqa

from nlpia.data.loaders import get_data

from nltk.sentiment import SentimentIntensityAnalyzer
from nlpia.models import LinearRegressor
from sklearn.linear_model import SGDRegressor


sms = get_data('sms-spam')
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs, columns=list(zip(*sorted([(v, k) for (k, v) in tfidf.vocabulary_.items()])))[1])

# TFIDF
tfidf_lda = LDA(n_components=1)
tfidf_lda.fit(tfidf_docs, sms.spam)
# UserWarning: Variables are collinear. warnings.warn("Variables are collinear.")
sms['tfidf_lda_spam_prob'] = tfidf_lda.predict_proba(tfidf_docs)[:, 1]
# Almost all 00000...0001 or .9999999...

# TFIDF->PCA
pca = PCA(n_components=256)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=['topic{}'.format(i) for i in range(pca.n_components_)])
pca_components = pd.DataFrame(pca.components_,
                              columns=tfidf_docs.columns,
                              index=['topic{}'.format(i) for i in range(pca.n_components_)])
# TFIDF->PCA->LDA
pca_lda = LDA(n_components=1)
pca_lda.fit(pca_topic_vectors, sms.spam)
sms['pca_lda_spam_prob'] = pca_lda.predict_proba(pca_topic_vectors)[:, 1]


# Most spammy topics (based on correlation of LSA/PCA topics with spam_labels)
pca_topic_vectors['spam_label_'] = sms.spam
print(pca_topic_vectors.corr().spam_label_.sort_values(ascending=False).head())
# spam_label_    1.000000
# topic4         0.564850
# topic2         0.275897
# topic10        0.186002
# topic9         0.167077
# Name: spam_label_, dtype: float64

# Least spammy topics from PCA
print(pca_topic_vectors.corr().spam_label_.sort_values().head())
# topic0    -0.220491
# topic18   -0.147774
# topic1    -0.146163
# topic3    -0.125849
# topic28   -0.113993
# Name: spam_label_, dtype: float64


# Most spammy terms (words) in TFIDF:
tfidf_lda_coef = pd.DataFrame(list(zip(tfidf_lda.coef_[0, :], tfidf_docs.columns)), columns='coef term'.split())
print(tfidf_lda_coef.sort_values('coef', ascending=False).head())
#               coef                 term
# 2666  7.606693e+06              darling
# 7168  5.393772e+06             sexychat
# 895   5.202198e+06                80488
# 6085  4.865422e+06              parties
# 9025  4.852177e+06  www.07781482378.com

# Least spammy terms (words) in TFIDF
print(tfidf_lda_coef.sort_values('coef', ascending=True).head())
#               coef       term
# 1320 -2.751719e+06      anjie
# 7108 -2.751719e+06        sef
# 6673 -2.658972e+06    reapply
# 6836 -2.658972e+06   resuming
# 5323 -2.633044e+06  memorable

# Not very interesting
# sms.plot.scatter(x='tfidf_lda_spam_prob', y='pca_lda_spam_prob')
# plt.tight_layout()
# plt.show()

vader = SentimentIntensityAnalyzer()
scores = pd.DataFrame([vader.polarity_scores(text) for text in sms.text])
sms['vader'] = scores['compound']
mask = sms.vader != 0

line = LinearRegressor()
line = line.fit(pca_topic_vectors['topic4'], sms['vader'])
sms['line'] = line.predict(pca_topic_vectors['topic4'])

sgd = SGDRegressor(n_iter=200)
sgd = sgd.fit(pca_topic_vectors[['topic4']], scores['compound'])
sms['sgd'] = sgd.predict(pca_topic_vectors[['topic4']])

sms['pca_lda_spaminess'] = 2 * sms.pca_lda_spam_prob - 1


class OneNeuronRegressor:

    def __init__(self, n_inputs=1, n_iter=1000, alpha=0.1):
        self.n_inputs = n_inputs
        self.n_outputs = 1
        self.W1 = np.random.randn(self.n_outputs, self.n_inputs + 1)
        self.n_iter = n_iter
        self.alpha = alpha

    def error(self, X, y):
        # Calculate predictions (forward propagation)
        z1 = self.predict(y.reshape(len(X), 1))
        return (y - z1)

    def delta(self, X, y):
        e = self.error(X, y)
        deltaW1 = X.T.dot(e)
        # deltaW1 = np.array([np.sum(deltaW1[:, i]) for i in range(self.W1.shape[0])]).reshape(self.W1.shape) / len(X)
        return deltaW1

    def fit(self, X, y, n_iter=None):
        self.n_iter = self.n_iter if n_iter is None else n_iter
        for i in range(self.n_iter):
            deltaW1 = self.delta(X, y)
            self.W1[0, 0] = self.W1[0, 0] + self.alpha * deltaW1
            self.W1[0, 1] = self.W1[0, 1] + self.alpha * np.mean(self.error(X, y))

            # self.b1 += self.alpha * deltab1
        return self

    def predict(self, X):
        X1 = np.ones((len(X), self.n_inputs + 1))
        X1[:, 0] = X[:, 0]
        return self.W1.dot(X1.T).T


X = pca_topic_vectors[['topic4']].values[:5, :]
y = scores['compound'].reshape(len(scores), 1).values[:5, :]

nn = OneNeuronRegressor(n_iter=1)
for i in range(3):
    print('-' * 10)
    print(nn.W1)
    print(nn.predict(X))
    print(nn.error(X, y))
    print(nn.delta(X, y))
    nn = nn.fit(X, y, 1)
    print(nn.W1)
    print(pd.DataFrame(nn.error(X, y)).mean())[0]

nn = nn.fit(pca_topic_vectors[['topic4']], scores['compound'].reshape(len(scores), 1), 1000)

sms['nn'] = nn.predict(pca_topic_vectors[['topic4']])


def sentiment_scatter(sms=sms):
    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(1, 1, 1)
    ax = sms.plot.scatter(x='topic4', y='line', ax=ax, color='g', marker='+', alpha=.6)
    ax = sms.plot.scatter(x='topic4', y='sgd', ax=ax, color='r', marker='x', alpha=.4)
    ax = sms.plot.scatter(x='topic4', y='vader', ax=ax, color='k', marker='.', alpha=.3)
    ax = sms.plot.scatter(x='topic4', y='sgd', ax=ax, color='c', marker='s', alpha=.6)
    ax = sms.plot.scatter(x='topic4', y='pca_lda_spaminess', ax=ax, color='b', marker='o', alpha=.6)
    plt.ylabel('Sentiment')
    plt.xlabel('Topic 4')
    plt.legend(['LinearRegressor', 'SGDRegressor', 'Vader', 'OneNeuronRegresor', 'PCA->LDA->spaminess'])
    plt.tight_layout()
    plt.show()


sentiment_scatter()
