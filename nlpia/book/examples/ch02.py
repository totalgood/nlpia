# Chapter 2 Code Examples


sentence = "Thomas Jefferson began building Monticello at the age of twenty-six."
sentence.split()
# ['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', 'twenty-six.']

# As you can see, this simple Python function already does a decent job tokenizing the example sentence. A couple more vanilla python statements and you can create numerical vector representations for each word.

sorted(dict([(token, 1) for token in sentence.split()]).items())
[('Jefferson', 1),
 ('Monticello', 1),
 ('Thomas', 1),
 ('age', 1),
 ('at', 1),
 ('began', 1),
 ('building', 1),
 ('of', 1),
 ('the', 1),
 ('twenty-six.', 1)]


# A slightly better data structure

import pandas as pd
sentence = "Thomas Jefferson began building Monticello at the age of 26."
df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T
df
#    26.  Jefferson  Monticello  Thomas  age  at  began  building  of  the
# 0    1          1           1       1    1   1      1         1   1    1


# And a pandas dataframe is great for holding multiple texts (sentences, tweets, or documents)
sentences = "Construction was done mostly by local masons and carpenters.\n" \
            "He moved into the South Pavilion in 1770.\n" \
            "Turning Monticello into a neoclassical masterpiece in the Palladian style was his perennial project.\n" \
for i, sent in enumerate(sentences.split('\n')):
    df['sent{}'.format(i)] = dict([token, 1 for token in sent.split()])

#       26.  Jefferson  Monticello  Thomas  age  at  began  building  of  the
# sent    1          1           1       1    1   1      1         1   1    1


# And a pandas dataframe is great for holding multiple texts (sentences, tweets, or documents)
sentences = "Construction was done mostly by local masons and carpenters.\n" \
            "He moved into the South Pavilion in 1770.\n" \
            "Turning Monticello into a neoclassical masterpiece in the Palladian style was his perennial project.\n"
for i, sent in enumerate(sentences.split('\n')):
    df['sent{}'.format(i)] = dict([token, 1 for token in sent.split()])

