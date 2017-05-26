# Chapter 3 examples

from collections import Counter

import pandas as pd
from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D

from nltk.tokenize import TreebankWordTokenizer

sentence = "The faster Harry got to the store, the faster and faster Harry would get home."
tokenizer = TreebankWordTokenizer()
tokenize = tokenizer.tokenize
token_sequence = tokenize(sentence)

lexicon = ['faster', 'Harry', 'home']
vector1 = Counter(tok for tok in tokenize("The faster Harry got to the store, the faster and faster Harry would get home.") if tok in lexicon)
vector2 = Counter(tok for tok in tokenize("Jill is faster than Harry.") if tok in lexicon)
vector3 = Counter(tok for tok in tokenize("Jill and Harry fast.") if tok in lexicon)
corpus = [vector1, vector2, vector3]

corpus
# [Counter({'Harry': 2, 'faster': 3, 'home': 1}),
#  Counter({'Harry': 1, 'faster': 1}),
#  Counter()]

df = pd.DataFrame.from_records(corpus)
df = df.fillna(0)
df
#    Harry  faster  home
# 0    2.0     3.0   1.0
# 1    1.0     1.0   0.0
# 2    1.0     0.0   0.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(df.Harry, df.faster, df.home, s=20 * df.T.sum(), c=list('rgb'))
plt.xlabel('Harry')
plt.ylabel('faster')
ax.set_zlabel('home')
plt.show()
