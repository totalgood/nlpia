#!/usr/bin/env python
r""" Chapter 3 Zipf City Population plot by Krugman using Wikipedia data

>>> df = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population', header=0)[3]
>>> rankpop = df[[df.columns[0], df.columns[3]]]
>>> axes = rankpop.plot(style='o', logx=True, logy=True)
>>> xlabel = plt.xlabel('US City Rank in Population')
>>> ylabel = plt.ylabel('US City Population')
>>> plt.title('Counting People in Cities Like Counting Words in Documents')
>>> plt.tight_layout()
>>> plt.show()
"""
import pandas as pd
import seaborn  # noqa
from matplotlib import pyplot as plt

from nlpia.loaders import clean_columns


dfs = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population', header=0)
df = dfs[4]
df.to_csv
df.columns = clean_columns(df.columns)
rank_col = next(c for c in df.columns if 'rank' in c)
rank = df[rank_col]
pop_col = next(c for c in df.columns if 'est' in c)
population = df[pop_col]
rankpop = pd.DataFrame(population)
rankpop.index = rank
rankpop.columns = ['Population']
fig = plt.figure()
ax = fig.add_subplot(111)
rankpop.plot(style='o', logx=True, logy=True, ax=ax)

xlabel = plt.xlabel('Rank')
ylabel = plt.ylabel('US City Population')
plt.show(block=False)
# plt.title('People Counts in Cities Like Word Counts in Documents')
# plt.tight_layout()
# plt.show(block=False)


r"""
>>> nltk.download('brown')  # <1>
>>> from nltk.corpus import brown
>>> brown.words()[:10]  # <2>
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of']
>>> brown.tagged_words()[:5]  # <3>
[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL')]
>>> len(brown.words())
1161192
"""
import nltk  # noqa
nltk.download('brown')  # <1>
from nltk.corpus import brown  # noqa
from collections import Counter  # noqa

r"""
>>> from collections import Counter
>>> puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
>>> word_list = (x.lower() for x in brown.words() if x not in puncs and '_' not in x and ' ' not in x)
>>> token_counts = Counter(word_list)
>>> token_counts['token'] = token_counts.index
>>> token_counts.index = range(1, len(token_counts) + 1)
>> token_counts.most_common(20)  # builtin method of a Counter Object
"""
puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
word_generator = (x.lower() for x in brown.words() if x not in puncs and '_' not in x and ' ' not in x)
token_counts = pd.DataFrame(((w, c) for (w, c) in Counter(word_generator).items() if c > 10), columns=['Token', 'Token Count'])
token_counts.sort_values('Token Count', ascending=False)
# token_counts['Token'] = token_counts.index
# token_counts.index = range(1, len(token_counts) + 1)
# token_counts[['Token Count']].plot(style='.', logx=True, logy=True, ax=ax, secondary_y=True)
token_counts[['Token Count']].plot(style='+', logx=True, logy=True, ax=ax)
xlabel = plt.xlabel('Rank')
ylabel = plt.ylabel('Word Count')
plt.show(block=False)
# plt.title('People Counts in Cities Like Word Counts in Documents')
# plt.tight_layout()
# plt.show(block=False)
