""" Chapter 3 Zipf City Population plot by Krugman using Wikipedia data
>>> df = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population', header=0)[3]
>>> rankpop = df[[df.columns[0], df.columns[3]]]
>>> axes = rankpop.plot(style='o', logx=True, logy=True)
>>> xlabel = plt.xlabel('US City Rank in Population')
>>> ylabel = plt.ylabel('US City Population')
>>> title plt.title('Counting People in Cities Like Counting Words in Documents')
>>> plt.tight_layout()
>>> plt.show()
"""
import pandas as pd

df = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population', header=0)[3]
rankpop = df[[df.columns[0], df.columns[3]]]
axes = rankpop.plot(style='o', logx=True, logy=True)
xlabel = plt.xlabel('US City Rank in Population')
ylabel = plt.ylabel('US City Population')
title plt.title('Counting People in Cities Like Counting Words in Documents')
plt.tight_layout()
plt.show()