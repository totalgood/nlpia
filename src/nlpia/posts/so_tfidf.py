""" Answer for https://stackoverflow.com/questions/35757560 """
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


corpus = ["The dog ate a sandwich and I ate a sandwich",
          "The wizard transfigured a sandwich"]
vectorizer = TfidfVectorizer(stop_words='english')
tfidfs = vectorizer.fit_transform(corpus)


columns = [k for (v, k) in sorted((v, k) for k, v in vectorizer.vocabulary_.items())]
tfidfs = pd.DataFrame(tfidfs.todense(), columns=columns)
#      ate   dog  sandwich  transfigured  wizard
# 0   0.75  0.38      0.54          0.00    0.00
# 1   0.00  0.00      0.45          0.63    0.63

df = (1 / pd.DataFrame([vectorizer.idf_], columns=columns))
#      ate   dog  sandwich  transfigured  wizard
# 0   0.71  0.71       1.0          0.71    0.71
corp = [txt.lower().split() for txt in corpus]
corp = [[w for w in d if w in vectorizer.vocabulary_] for d in corp]
tfs = pd.DataFrame([Counter(d) for d in corp]).fillna(0).astype(int)
#    ate  dog  sandwich  transfigured  wizard
# 0    2    1         2             0       0
# 1    0    0         1             1       1

# The first document's TFIDF vector:
tfidf0 = tfs.iloc[0] * (1. / df)
tfidf0 = tfidf0 / pd.np.linalg.norm(tfidf0)
#         ate       dog  sandwich  transfigured  wizard
# 0  0.754584  0.377292  0.536893           0.0     0.0

tfidf1 = tfs.iloc[1] * (1. / df)
tfidf1 = tfidf1 / pd.np.linalg.norm(tfidf1)
#     ate  dog  sandwich  transfigured    wizard
# 0   0.0  0.0  0.449436      0.631667  0.631667
