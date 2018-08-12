In writing this book we pulled from numerous resources. Here are some of our favorites.

In an ideal world, you could find these resources yourself simply by entering the heading text into a semantic search engine like [Duck Duck Go](http://duckduckgo.com), [Gigablast](http://gigablast.com/search?c=main&q=open+source+search+engine) or [Qwant](https://www.qwant.com/web). But until Jimmy Wales takes another shot at [Wikia Search](https://en.wikipedia.org/wiki/Wikia_Search) or Google shares their NLP technology, we have to rely on 90’s-style lists of links like this. Check out [section\_title](#search_engines_section) resources below if your contribution to saving the world includes helping open source projects that index the web.

Applications and Project Ideas
==============================

Here are some applications to inspire your own NLP projects.

-   [Guess passwords from Social Network profiles](http://www.sciencemag.org/news/2017/09/artificial-intelligence-just-made-guessing-your-password-whole-lot-easier)[1]

-   [Parking ticket lawyer bots](https://www.theguardian.com/technology/2016/jun/28/chatbot-ai-lawyer-donotpay-parking-tickets-london-new-york)[2] — Chatbots can file parking ticket legal appeals for you in New York and London

-   [Gutenberg + Library of Congress](https://github.com/craigboman/gutenberg)[3] — Automatic document classification according to the Library of Congress specification

-   [Longitudial Detection of Dementia Through Lexical and Syntactic Changes in Writing](ftp://ftp.cs.toronto.edu/dist/gh/Le-MSc-2010.pdf:)[4] — Masters thesis by Xian Le on psychology diagnosis with NLP

-   [Time Series Matching](https://www.cs.nyu.edu/web/Research/Theses/wang_zhihua.pdf)[5] — Songs and other time series can be discretized and searched with dynamic programming algorithms analogous to Levenshtein Distance

-   [NELL, Never Ending Learning](http://rtw.ml.cmu.edu/rtw/publications)[6] — publications for NELL, a constantly evolving knowledge base that learns by scraping natural language text

-   [How the NSA identified Satoshi Nakamoto](https://medium.com/cryptomuse/how-the-nsa-caught-satoshi-nakamoto-868affcef595) — Wired Magazine and the NSA identified Satoshi Nakamoto using NLP (this blog called it "stylometry")

-   [Stylometry](https://en.wikipedia.org/wiki/Stylometry) and [Natural Language Forensics](http://www.parkjonghyuk.net/lecture/2017-2nd-lecture/forensic/s8.pdf) — Style/pattern matching and clustering of natural language text (also music and artworks) for authoriship and attribution

-   Scrape [examples.yourdictionary.com](http://examples.yourdictionary.com/) for examples of various grammatically correct sentences with POS labels and train your own Parsey McParseface syntax tree and POS tagger.

-   [Identifying "Fake News" With NLP](https://nycdatascience.com/blog/student-works/identifying-fake-news-nlp/) by Julia Goldstein and Mike Ghoul at NYC Data Science Academy.

-   [simpleNumericalFactChecker](https://github.com/uclmr/simpleNumericalFactChecker) by [Andreas Vlachos](https://github.com/andreasvlachos)) and information extraction (see Chapter 11) could be used to rank publishers, authors, and reporters for truthfulness. Might be combined with "fake news" labeler above.

-   [artificial-adversary](https://github.com/airbnb/artificial-adversary) by Jack Dai, an intern at Airbnb — Obfuscates natural language text (e.g. 'ur gr8' ⇒ 'you are great'). You could train a machine learning classifier to detect this obfuscation. You could also train a stemmer (an autoencoder with the obfuscator generating character features) to decypher obfuscated words so your NLP pipeline can handle obfuscated text without retraining. Thank you Aleck.

Courses and Tutorials
=====================

Here are some good tutorials, demonstrations, and even courseware from renowned university programs, many of which include Python examples.

-   [Speech and Language Processing](https://web.stanford.edu/\~jurafsky/slp3/ed3book.pdf) by David Jurafsky and James H. Martin — The next book you should read if you’re serious about NLP. Jurafsky and Martin are more thorough and rigorous in their explanation of NLP concepts. They have whole chapters on topcis that we largely ignore, like Finite State Transducers (FSTs), Hidden Markhov Models (HMMs), Part-of-Speech (POS) tagging, syntactic parsing, discourse coherence, machine translation, summarization, and dialog systems.

-   [MIT Artificial General Intelligence Course (6.S099)](https://agi.mit.edu)[7] led by Lex Fridman Feb 2018 — MIT’s free, interactive (there’s a public competition) AGI course. It’s probably the most thorough and rigorous course on engineering Artificial Intelligence engineering you can find.

-   [Textacy](https://github.com/chartbeat-labs/textacy)[8] — Topic modeling wrapper for SpaCy

-   [MIT Natural Language and the Computer Representation of Knowledge](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003/lecture-notes/)[9]

-   [SVD](http://people.revoledu.com/kardi/tutorial/LinearAlgebra/SVD.html)[10] — Singular Value Decomposition by Kardi Teknomo, PhD.

-   [Intro to IR (and NLP)](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)[11] — *An Introduction to Information Retrieval*, April 1, Online Edition, by Christopher Manning, Prabhakar Raghavan, and Hinrich Schutze at Stanford (creators of the Stanford CoreNLP library)

Research Papers and Talks
=========================

One of the best way to gain a deep understanding of a topic is to try to repeat the experiments of researchers and then modify them in some way. That’s how the best professors and mentors "teach" their students, by just encouraging them to try to duplicate the results of other researchers they are interested in. You can’t help but tweak an approach if you spend enough time trying to get it to work for you.

Vector Space Models and Semantic Search
---------------------------------------

-   [Semantic Vector Encoding and Similarity Search Using Full Text Search Engines](https://arxiv.org/pdf/1706.00957.pdf) — Jan Rygl et. al were able to use a conventional inverted index to implement efficient semantic search for all of Wikipedia.

-   [Learning Low-Dimensional Metrics](https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics.pdf) — Lalit Jain, et al, were able to incorporate human judgement into pairwise distance metrics which can be used for better decision-making and unsupervised clustering of word vectors and topic vectors. For example recruiters can use this to steer a content-based recommendation engine that matches resumes with job descriptions.

-   [RAND-WALK: A latent variable model approach to word embeddings](https://arxiv.org/pdf/1502.03520.pdf) by Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, Andrej Risteski — Explains the latest (2016) understanding of the "vector oriented reasoning" of Word2vec and other word vector space models, particular analogy questions.

-   [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Tomas Mikolov, Greg Corrado, Kai Chen, Jeffrey Dean at Google, Sep 2013 — first publication of the Word2vec model, including an implementation in C++ and pretrained models using a Google News corpus

-   [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean at Google — Describes refinements to the word2vec model that improved its accuracy, including subsampling and negative sampling.

Finance
-------

-   [Predicting Stock Returns by Automatically Analyzing Company News Announcements](http://www.stagirit.org/sites/default/files/articles/a_0275_ssrn-id2684558.pdf) — Bella Dubrov used gensim’s Doc2vec to predict stock prices based on company announcements with excellent explanations of `Word2vec` and `Doc2vec`.

-   [Building a Quantitative Trading Strategy To Beat the S&P500](https://www.youtube.com/watch?v=ll6Tq-wTXXw) — At PyCon 2016 Karen Rubin explained how she discovered that Female CEOs are predictive of rising stock prices, though not as strongly as she initially thought

Question Answering Systems
--------------------------

-   [Visual question answering](https://github.com/avisingh599/visual-qa)[12]

-   [2005 tutorial](http://lml.bas.bg/ranlp2005/tutorials/magnini.ppt)[13]

-   [2003 EACL tutorial by Lin Katz, University of Waterloo, Canada](https://cs.uwaterloo.ca/\~jimmylin/publications/Lin_Katz_EACL2003_tutorial.pdf)[14]

-   [visual question answering](https://github.com/avisingh599/visual-qa)[15]

-   [simple question answering from scratch (`corenlp` and `nltk` used for sentence segmenting and POS tagging)](https://github.com/raoariel/NLP-Question-Answer-System/blob/master/simpleQueryAnswering.py)[16]

-   [Attardi et al, 2001 "PiQASso: Pisa Question Answering System"](http://trec.nist.gov/pubs/trec10/papers/piqasso.pdf) uses traditional Information Retrieval (IR) NLP.[17]

Competitions and Awards
=======================

-   [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html) — Some researchers believe that compression of natural language text is equivalent to Artificial General Intelligence (AGI)

-   [The Hutter Prize](https://en.wikipedia.org/wiki/Hutter_Prize) — Annual competition to compress a 100 MB archive of Wikipedia natural language text. Alexander Rhatushnyak won in 2017.

-   [Open Knowledge Extraction Challenge](https://svn.aksw.org/papers/2017/ESWC_Challenge_OKE/public.pdf)

Datasets
========

Natural language data is everywhere you look. Language is the superpower of the human race, and your pipeline should take advantage of it.

-   [Stanford Datasets](https://nlp.stanford.edu/data/) — Pretrained `word2vec` and GloVE models, multilingual language models and datasets, multilingual dictionaries, lexica and corpora.

-   [Pretrained word2vec models](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-model) — The README for a word vector web API provides links to several word vector models, including the 300D Wikipedia GloVE model

-   [Some NLP Datasets](https://github.com/karthikncode/nlp-datasets)

-   [A LOT of NLP Datasets](https://github.com/niderhoff/nlp-datasets)

-   [Google’s International TTS](https://github.com/googlei18n/language-resources) — Data and tools for i18n

-   [Natural Language Processing in Action](https://github.com/totalgood/nlpia) —  Python package with data loaders and preprocessors for all the NLP data you will ever need…​ until you finish this book ;)

Search Engines
==============

Search Algorithms
-----------------

-   [GPU-enhanced BidMACH](https://arxiv.org/pdf/1702.08734.pdf) — BidMACH is a highdeminsional vector indexing and KNN search implementation, similar to the `annoy` python package. This paper explains an enhancement for GPUs that is 8x faster than the original implementation.

-   [Erik Bernhardsson’s Annoy Package](https://erikbern.com/2017/11/26/annoy-1.10-released-with-hamming-distance-and-windows-support.html) — Erik built the annoy nearest neighbor algorithm at Spotify and continues to enhance it.

-   [Erik Bernhardsson’s ANN Comparison](https://erikbern.com/2018/02/15/new-benchmarks-for-approximate-nearest-neighbors.html) — Approximate Nearest Neighbor algorithms are the key to scalable semantic search, and Erik keeps the world abreast of the latest and greatest algorithms out there.

Open Source Search Engines
--------------------------

-   [BeeSeek](https://launchpad.net/\~beeseek-devs) — Open source distributed web indexing and private search (hive search). No longer maintained.

-   [WebSphinx](https://www.cs.cmu.edu/\~rcm/websphinx/) — Web GUI for building a web crawler

Open Source Full Text Indexers
------------------------------

Efficient indexing is critical to any natural language search application. Here are a few open source full text indexing options. However, these "search engines" do not crawl the web, so you need to provide them with the corpus you want them to index and search.

-   [Elastic Search](https://github.com/elastic/elasticsearch)

-   [Apache Solr](https://github.com/apache/lucene-solr)

-   [Sphinx Search](https://github.com/sphinxsearch/sphinx)

-   [Xapian](https://github.com/Kronuz/Xapiand) — There are packages for Ubuntu that will let you search your local hard drive (like Google Desktop used to do)

-   [Indri](http://www.lemurproject.org/indri.php) — Semantic search with a [Python interface](https://github.com/cvangysel/pyndri) but it doesn’t seem to be actively maintained.

-   [Gigablast](https://github.com/gigablast/open-source-search-engine) — Open source web crawler and natural language indexer in C++.

-   [Zettair](http://www.seg.rmit.edu.au/zettair) open source HTML and TREC indexer (no crawler or live example). Last updated 2009.

-   [OpenFTS](http://openfts.sourceforge.net) — Full text search indexer for PostgreSQL with Python API [PyFTS](http://rhodesmill.org/brandon/projects/pyfts.html).

Manipulative Search Engines
---------------------------

The search engines most of us use are not optimized solely to help you find what you need, but rather to ensure that you click on links that generate revenue for the company that built it. Google’s innovative second-price sealed-bid auction ensures that advertisers don’t overpay for their ads,[18] but does not prevent search users from overpaying when they click on disguised advertisements. This manipulative search is not unique to Google, but any search engine that ranks results according to any other "objective function" other than your satisfaction with the search results. But here they are, if you want to compare and experiment.

-   google.com

-   bing.com

-   baidu.com

Less Manipulative Search Engines
--------------------------------

To determine how "commercial" and manipulative a search engine was, I queried several engines with things like "open source search engine". I then counted the number of ad-words purchasers and click-bait sites were among the search results on in the top 10. The sites below kept that count below one or two. And the top search results were often the most objective and useful sites, like Wikipedia, Stack Exchange, or reputable news articles and blogs.

-   [Alternatives to Google](https://www.lifehack.org/374487/try-these-15-search-engines-instead-google-for-better-search-results)

-   [Yandex.com](https://yandex.com/search/?text=open%20source%20search%20engine&lr=21754) — Surprisingly, the most popular Russian search engine (60% of Russian searches) seemed less manipulative than the top US search engines.

-   [DuckDuckGo.com](https://duckduckgo.com)

-   [`watson` Semantic Web Search](http://watson.kmi.open.ac.uk/WatsonWUI) — No longer in development, and not really a full text web search, but it is an interesting way to explore the semantic web (at least what it was years ago before `watson` was frozen)

Distributed Search Engines
--------------------------

Distributed search engines,[19][20] are perhaps the least manipulative and most "objective" because they have no central server to influence the ranking of the search results. However, current distributed search implementations rely on TF-IDF word frequencies to rank pages, because of the difficulty in scaling and distributing semantic search NLP algorithms. However, distribution of semantic indexing approaches like Latent Semantic Analysis (LSA) and Locality Sensitive Hashing have been successfully distributed with nearly linear scaling (as good as you can get). It’s just a matter of time before someone decides to contribute code for semantic search into an open source project like Yacy or builds a new distributed search engine capable of LSA.

-   [Nutch](https://nutch.apache.org/) — Nutch spawned Hadoop and itself became less of a distributed search engine and more of a distributed HPC system over time

-   [Yacy](https://www.yacy.net/en/index.html) — One of the few [open source](https://github.com/yacy/yacy_search_server) decentralized (at least federated) search engines and web crawlers still actively in use. Preconfigured clients for Mac, Linux, and Windows are available.

[1] [sciencemag.org/news/2017/09/artificial-intelligence-just-made-guessing-your-password-whole-lot-easier](http://www.sciencemag.org/news/2017/09/artificial-intelligence-just-made-guessing-your-password-whole-lot-easier)

[2] [theguardian.com/technology/2016/jun/28/chatbot-ai-lawyer-donotpay-parking-tickets-london-new-york](https://www.theguardian.com/technology/2016/jun/28/chatbot-ai-lawyer-donotpay-parking-tickets-london-new-york)

[3] [github.com/craigboman/gutenberg](https://github.com/craigboman/gutenberg)

[4] [ftp.cs.toronto.edu/dist/gh/Le-MSc-2010.pdf](ftp://ftp.cs.toronto.edu/dist/gh/Le-MSc-2010.pdf)

[5] [cs.nyu.edu/web/Research/Theses/wang\_zhihua.pdf](https://www.cs.nyu.edu/web/Research/Theses/wang_zhihua.pdf)

[6] [rtw.ml.cmu.edu/rtw/publications](http://rtw.ml.cmu.edu/rtw/publications)

[7] [agi.mit.edu](https://agi.mit.edu)

[8] [github.com/chartbeat-labs/textacy](https://github.com/chartbeat-labs/textacy)

[9] [ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003/lecture-notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003/lecture-notes/)

[10] [people.revoledu.com/kardi/tutorial/LinearAlgebra/SVD.html](http://people.revoledu.com/kardi/tutorial/LinearAlgebra/SVD.html)

[11] [nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)

[12] [github.com/avisingh599/visual-qa](https://github.com/avisingh599/visual-qa)

[13] [lml.bas.bg/ranlp2005/tutorials/magnini.ppt](http://lml.bas.bg/ranlp2005/tutorials/magnini.ppt)

[14] [cs.uwaterloo.ca/~jimmylin/publications/Lin\_Katz\_EACL2003\_tutorial.pdf](https://cs.uwaterloo.ca/~jimmylin/publications/Lin_Katz_EACL2003_tutorial.pdf)

[15] [github.com/avisingh599/visual-qa](https://github.com/avisingh599/visual-qa)

[16] [github.com/raoariel/NLP-Question-Answer-System/blob/master/simpleQueryAnswering.py](https://github.com/raoariel/NLP-Question-Answer-System/blob/master/simpleQueryAnswering.py)

[17] [trec.nist.gov/pubs/trec10/papers/piqasso.pdf](http://trec.nist.gov/pubs/trec10/papers/piqasso.pdf)

[18] Cornell University Networks Course case study, "Google Adwords Auction", [blogs.cornell.edu/info2040/2012/10/27/google-adwords-auction-a-second-price-sealed-bid-auction](https://blogs.cornell.edu/info2040/2012/10/27/google-adwords-auction-a-second-price-sealed-bid-auction/)

[19] [en.wikipedia.org/wiki/Distributed\_search\_engine](https://en.wikipedia.org/wiki/Distributed_search_engine)

[20] [wiki.p2pfoundation.net/Distributed\_Search\_Engines](https://wiki.p2pfoundation.net/Distributed_Search_Engines)
