# Chapter 3 Code Examples

```python
>>> sentence = "Thomas Jefferson began building Monticello at the age of twenty-six."
>>> sentence.split()
['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', 'twenty-six.']
```

As you can see, this simple Python function already does a decent job tokenizing the example sentence. A couple more vanilla python statements and you can create numerical vector representations for each word.

```python
>>> binary_vector = sorted(dict([(token, 1) for token in sentence.split()]).iteritems())
```

```python
>>> import pandas as pd
>>> sentence = "Thomas Jefferson began building Monticello at the age of 26."
>>> df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()])), columns=['sent']).T
>>> df
      26.  Jefferson  Monticello  Thomas  age  at  began  building  of  the
sent    1          1           1       1    1   1      1         1   1    1
```

```python
>>> sentences = "Construction was done mostly by local masons and carpenters.\n" \
...             "He moved into the South Pavilion in 1770.\n" \
...             "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."
>>> corpus = {'sent0': df.T['sent'].to_dict()}
>>> for i, sent in enumerate(sentences.split('\n')):
...     corpus['sent{}'.format(i + 1)] = dict((tok, 1) for tok in sent.split())
>>> df = pd.DataFrame(corpus, dtype=int).fillna(0)
>>> df.head(10)  # show the first 10 tokens in our vocabulary for this 4-document corpus
              sent0  sent1  sent2  sent3
1770.           0.0    0.0    1.0    0.0
26.             1.0    0.0    0.0    0.0
Construction    0.0    1.0    0.0    0.0
He              0.0    0.0    1.0    0.0
Jefferson       1.0    0.0    0.0    0.0
Jefferson's     0.0    0.0    0.0    1.0
Monticello      1.0    0.0    0.0    1.0
Pavilion        0.0    0.0    1.0    0.0
South           0.0    0.0    1.0    0.0
Thomas          1.0    0.0    0.0    0.0
```

```python
Turning         0.0    0.0    0.0    1.0
a               0.0    0.0    0.0    1.0
age             1.0    0.0    0.0    0.0
and             0.0    1.0    0.0    0.0
at              1.0    0.0    0.0    0.0
began           1.0    0.0    0.0    0.0
building        1.0    0.0    0.0    0.0
by              0.0    1.0    0.0    0.0
carpenters.     0.0    1.0    0.0    0.0
done            0.0    1.0    0.0    0.0
in              0.0    0.0    1.0    0.0
into            0.0    0.0    1.0    1.0
local           0.0    1.0    0.0    0.0
masons          0.0    1.0    0.0    0.0
masterpiece     0.0    0.0    0.0    1.0
mostly          0.0    1.0    0.0    0.0
moved           0.0    0.0    1.0    0.0
neoclassical    0.0    0.0    0.0    1.0
obsession.      0.0    0.0    0.0    1.0
of              1.0    0.0    0.0    0.0
the             1.0    0.0    1.0    0.0
was             0.0    1.0    0.0    1.0
```