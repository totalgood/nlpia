# Environment Software Requirements (Dependencies)

Here are the tools you need. They're first listed individually with their dependencies (thank you [`pipdeptree`](https://github.com/naiquevin/pipdeptree)!). Then we show just the leaves of a collapsed dependecy tree. Not only will this show you just the minimum set of things you need to install (because they'll install their dependencies automatically) and it will start to get you familiar with the concept of a Dependence Tree, or Directed Acyclic Graph (DAG). We'll see more of these when we discuss Syntax Net and other NLP syntax parsing tools.

* A Windows, Linux, or Mac PC
  ** 4 GB of RAM
  ** 10 GB of storage
* A http://www.github.com/[GitHub] account
  ** [`git` and a shell](https://git-scm.com/downloads) or [git for Windows](https://git-for-windows.github.io/)
* `python3` and some Python packages
  ** scikit-learn
    **** SciPy>=0.13.2: 
      ***** NumPy>=1.7.1
  ** gensim==0.13.3
    *** boto==2.43.0
    *** bz2file==0.98
    *** gensim==0.13.3
    *** numpy==1.11.2
    *** requests==2.11.1
    *** scipy>=0.18.1
      ***** NumPy>=1.7.1
    ***** six>=1.10.0
  ** seaborn==0.7.1
    *** NumPy>=1.8.0: efficient math
    *** SciPy>=0.13.2: 
      **** NumPy>=1.7.1
    *** matplotlib>=1.5.3
    *** pandas>=0.18.1
    *** statsmodels>=0.6.1
    *** patsy>=0.4.1
  ** TextBlob>=0.11.1
    *** nltk>=3.2.1
      **** NumPy>=1.8.0
      **** SciPy>=0.13.2
        ***** NumPy>=1.7.1
      **** matplotlib>=1.3.1
      **** scikit-learn>=0.14.1
      **** python-crfsuite>=0.8.2
      **** scikit-learn>=0.14.1
      **** gensim>=0.11.1
        ***** boto==2.43.0
        ***** bz2file==0.98
        ***** gensim==0.13.3
        ***** numpy==1.11.2
        ***** requests==2.11.1
        ***** scipy>=0.18.1
          ***** NumPy>=1.7.1
        ***** six>=1.10.0
        ***** smart-open>=1.3.5
      **** pyparsing>=2.0.3 for Lex/Yac grammars
      **** twython>=3.2.0
  ** ChatterBot
    *** Django
    *** fuzzywuzzy>=0.12,<0.13
      **** optional library: apt-get install python-Levenshtein
    *** jsondatabase>=0.1.1
    *** nltk<4.0.0
    *** pymongo>=3.3.0,<4.0.0
    *** python-twitter>=3.0
    *** textblob>=0.11.0,<0.12.0
  ** language-check
  ** tweepy
  ** DetectorMorse
    *** nlup
    *** jsonpickle

If you took the time to read that list, you might have noticed some inter-dependencies, loops, or cycles in that graph. But, ignoring version numbers and leaving the graph-walking to `pip`, you might see that the only things you really need to install are.

* gensim
* jupyter
* seaborn
* ChatterBot
* tweepy
* language-check

## Python Package Dependencies

[nltk](https://github.com/nltk/nltk/blob/develop/pip-req.txt)
[SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
[matplotlib](https://github.com/matplotlib/matplotlib)
[textblob](https://github.com/sloria/TextBlob)
[gensim](https://github.com/RaRe-Technologies/gensim)
[pyparsing](https://github.com/greghaskins/pyparsing/tree/master/src)
[seaborn](https://github.com/mwaskom/seaborn/blob/master/requirements.txt)
[ChatterBot](https://github.com/gunthercox/ChatterBot/blob/master/requirements.txt)
[syntaxnet](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html)

`stanford_corenlp_pywrapper` can be installed with:

```bash
git clone https://github.com/brendano/stanford_corenlp_pywrapper
cd stanford_corenlp_pywrapper
pip install .
```

## Binary Package Dependencies

### Optional

- [python-Levenshtein](https://github.com/seatgeek/fuzzywuzzy/issues/67)

## Java

[Stanford Core NLP Source Code](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip)
[Stanford Core NLP Jar](https://github.com/stanfordnlp/CoreNLP)

If you're able to install language-check or grammar-check you'll need this.

- [LanguageTool](http://www.languagetool.org/download/LanguageTool-stable.zip)

## Web APIs

[grammar parser](http://nlp.stanford.edu:8080/parser/index.jsp)




