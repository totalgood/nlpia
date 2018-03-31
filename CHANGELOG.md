<a name="0.1.21"></a>
# [0.1.21](https://github.com/totalgood/nlpia/compare/0.1.21...0.1.0) (2018-03-31)


### Bug Fixes

* installation instructions in README fixed
* pip install fixed and broken several times ;)
* some unittests fixed ;)
* fixed loaders for Word2Vec
* fixed chapter code for import of `loaders.get_data()` (duplicated module with `import *`)


### Features

* NLPIA Book examples in src/nlpia/
* upgraded pyscaffold to 3.0+ and folder structure (nlpia/nlpia -> nlpia/src/nlpia)
* fixed environment.yml
* added `requirements-*.txt` for various optional features
* loader and data/* files for simple "cats_and_dogs" corpora for LSA
* data/utf8.csv translation table
* utf-8 -> ascii translator/cleaner
* loaders can handle .txt, .csv, .csv.gz, and .json files automagically (just add them to data directory and shorten their file name)

<a name="0.1.0"></a>
# [0.1.0](https://github.com/totalgood/nlpia/compare/0.1.0...0.0.1) (2017-11~07)

Initial release for conference tutorial on building a Chatbot.
