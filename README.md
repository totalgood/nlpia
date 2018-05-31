# NLPIA

The community - driven code for [**N**atural ** L**anguage ** P**rocessing ** i**n ** A**ction](https: // bit.ly / nlpiabook).

# Description

A community - developed book about building Natural Language Processing pipelines for prosocial chatbots that contribute to communities.

# Getting Started

1. Install Python

[Anaconda](https: // docs.anaconda.com / anaconda / install /) or miniconda are easy ways to get python3 installed without messing up your system python on any machine (including Windows and Mac OSX). 
Another advantage of the `conda` package manager is that it can install some packages that fail with pip, due to binary dependencies like `QtPython` for `matplotlib` or `pyaudio` for `SpeechRecognition`.
 
```bash
export OS=Linux  # or OS=MacOSX
wget -c "https://repo.continuum.io/miniconda/Miniconda3-latest-${OS}-x86_64.sh" -o $HOME/install_miniconda3.sh
chmod +x $HOME/install_miniconda3.sh
cd $HOME
./install_miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
echo 'export PATH=$HOME/miniconda/bin:$PATH' >> $HOME/.bash_profile
bash $HOME/.bash_profile  # or just restart your terminal
conda install pip
```

2. Clone this repository

```bash
git clone https://github.com/totalgood/nlpia.git
cd nlpia
```

3. Use `conda - env` OR `pip` to install dependencies

Depending on your OS you may have better luck using conda to install the dependencies

### Use `conda-env`

The environment.yml file creates a conda environment called `nlpiaenv`

```bash
conda env create -n nlpiaenv -f conda/environment.yml
conda install pip  # to get the latest version of pip
source activate nlpiaenv
```

### Use `pip`

```bash
pip install --upgrade pip
mkvirtualenv nlpiaenv
source nlpiaenv/bin/activate
pip install -r requirements-test.txt
pip install -e .
pip install -r requirements-deep.txt
```

The chatbots(including TTS and STT audio drivers) that come with `nlpia` may not be compatible with Windows due to problems install `pycrypto`.
If you are on a Linux or Darwin(Mac OSX) system or want to try to help us debug the pycrypto problem feel free to install the chatbot requirements:

```bash
# pip install -r requirements-chat.txt
# pip install -r requirements-voice.txt
```

4. Activate this new environment

```bash
# source activate nlpia
```

5. Install an "editable" `nlpia` package in this conda environment(also called nlpia)

```bash
# pip install -e .
```

6. Check out the code examples from the book in `nlpia / nlpia / book / examples`

```bash
# cd nlpia/book/examples
# ls
```

# Contributing

Help your fellow readers by contributing to your shared code and knowledge.
Here are some ideas for a few features others might find handy.

## Glossary Compiler

Skeleton code and APIs that could be added to the https://github.com/totalgood/nlpia/blob/master/src/nlpia/transcoders.py:`transcoders.py` module.


```python


def find_acronym(text):
    """Find parenthetical noun phrases in a sentence and return the acronym/abbreviation/term as a pair of strings.

    >>> find_acronym('Support Vector Machine (SVM) are a great tool.')
    ('SVM', 'Support Vector Machine')
    """
    return (abbreviation, noun_phrase)


```

```python


def glossary_from_dict(dict, format='asciidoc'):
    """ Given a dict of word/acronym: definition compose a Glossary string in ASCIIDOC format """
    return text


```

```python


def glossary_from_file(path, format='asciidoc'):
    """ Given an asciidoc file path compose a Glossary string in ASCIIDOC format """
    return text


def glossary_from_dir(path, format='asciidoc'):
    """ Given an path to a directory of asciidoc files compose a Glossary string in ASCIIDOC format """
    return text


```

## Semantic Search

Use a parser to extract only natural language sentences and headings/titles from a list of lines/sentences from an asciidoc book like "Natural Language Processing in Action".
Use a sentence segmenter in https://github.com/totalgood/nlpia/blob/master/src/nlpia/transcoders.py:[nlpia.transcoders] to split a book, like _NLPIA_, into a list sentences.




