[![Build Status](https://travis-ci.org/totalgood/nlpia.svg?branch=master)](https://travis-ci.org/totalgood/nlpia) 
[![Coverage](https://codecov.io/gh/totalgood/nlpia/branch/master/graph/badge.svg)](https://codecov.io/gh/totalgood/nlpia) 
[![GitHub release](https://img.shields.io/github/release/totalgood/nlpia.svg)](https://github.com/totalgood/nlpia/releases/latest)
[![PyPI version](https://img.shields.io/pypi/pyversions/nlpia.svg)](https://pypi.org/project/nlpia/)

<!---
Some more badges from grakn.ai and search of github markdown files

### downloads are no longer provided by pypi: https://mail.python.org/pipermail/distutils-sig/2016-May/028986.html
[![PyPI Package monthly downloads](https://img.shields.io/pypi/dm/nlpia.svg?style=flat)](https://pypi.python.org/pypi/nlpia


[![GitHub release](https://img.shields.io/github/release/graknlabs/grakn.svg)](https://github.com/graknlabs/grakn/releases/latest)
[![Build Status](https://travis-ci.org/graknlabs/grakn.svg?branch=internal)](https://travis-ci.org/graknlabs/grakn)
[![Slack Status](http://totalgood.herokuapp.com/badge.svg)](https://totalgood.com/slack)
[![Stack Overflow][stackoverflow-shield]][stackoverflow-link]
[![Download count](https://img.shields.io/github/downloads/graknlabs/grakn/total.svg)](https://grakn.ai/download)
---
[![Static Bugs](https://sonarcloud.io/api/project_badges/measure?project=ai.grakn%3Agrakn&metric=bugs)](https://sonarcloud.io/dashboard?id=ai.grakn%3Agrakn)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=ai.grakn%3Agrakn&metric=code_smells)](https://sonarcloud.io/dashboard?id=ai.grakn%3Agrakn)
[![Duplicated Code](https://sonarcloud.io/api/project_badges/measure?project=ai.grakn%3Agrakn&metric=duplicated_lines_density)](https://sonarcloud.io/dashboard?id=ai.grakn%3Agrakn)

[stackoverflow-shield]: https://img.shields.io/badge/stackoverflow-grakn-blue.svg
[stackoverflow-link]: https://stackoverflow.com/questions/tagged/grakn
--->

# NLPIA

Community-driven code for the book [**N**atural **L**anguage **P**rocessing **i**n **A**ction](https://bit.ly/nlpiabook).

## Description

A community-developed book about building socially responsible NLP pipelines that give back to the communities they interact with.

## Getting Started

You'll need a bash shell on your machine. 
[Git](https://git-scm.com/downloads) has installers that include bash shell for all three major OSes. 

Once you have Git installed, launch a bash terminal. 
It will usually be found among your other applications with the name `git-bash`. 


1. Install [Anaconda3 (Python3.6)](https://docs.anaconda.com/anaconda/install/)

* [Linux](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh)
* [MacOSX](https://repo.anaconda.com/archive/Anaconda3-5.2.0-MacOSX-x86_64.pkg)
* [Windows](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe)

If you're installing Anaconda3 using a GUI, be sure to check the box that updates your PATH variable. 
Also, at the end, the Anaconda3 installer will ask if you want to install VSCode. 
Microsoft's VSCode is supposed to be an OK editor for Python so feel free to use it. 

2. Install an Editor

You can skip this step if you are happy using `jupyter notebook` or `VSCode` or the editor built into Anaconda3. 

I like [Sublime Text](https://www.sublimetext.com/3). 
It's a lot cleaner more mature. 
Plus it has more plugins written by individual developers like you.

3. Install Git and Bash

* Linux -- already installed
* MacOSX -- already installed
* [Windows](https://git-scm.com/downloads)

If you're on Linux or Mac OS, you're good to go. Just figure out how to launch a terminal and make sure you can run `ipython` or `jupyter notebook` in it. This is where you'll play around with your own NLP pipeline. 

On Windows you have a bit more work to do. Supposedly Windows 10 will let you install Ubuntu with a terminal and bash. But the terminal and shell that comes with [`git`](https://git-scm.com/downloads) is probably a safer bet. It's mained by a broader open source community.

4. Clone this repository

```bash
git clone https://github.com/totalgood/nlpia.git
```

5. Install `nlpia` 

You have two tools you can use to install `nlpia`:

5.1. `conda`  
5.2. `pip`  


### 5.1. `conda`

In most cases, conda will be able to install python packages faster and more reliably than pip, because packages like `python-levenshtein` require you to compile a C library during installation, and Windows doesn't have an installer that will "just work."

So use conda (part of the Anaconda package that we already installed) to create an environment called `nlpiaenv`:

```bash
cd nlpia  # make sure you're in the nlpia directory that contains `setup.py`
conda env create -n nlpiaenv -f conda/environment.yml
conda install pip  # to get the latest version of pip
pip install -e .
```

Whenever you want to be able to import or run any `nlpia` modules, you'll need to activate this conda environment first:

```bash
source activate nlpiaenv
python -c "print(import nlpia)"
```


Skip to Step 4 if you have successfully created and activated an environment containing the `nlpia` package.

### 5.2. `pip`

Linux-based OSes like Ubuntu and OSX come with C++ compilers built-in, so you may be able to install the dependencies using pip instead of `conda`. 
But if you're on Windows and you want to install packages, like `python-levenshtein` that need compiled C++ libraries, you'll need a compiler. 
Fortunately Microsoft still lets you [download a compiler for free](https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29), just make sure you follow the links to the Visual Studio "Build Tools" and not the entire Visual Studio package.

Once you have a compiler on your OS you can install `nlpia` using pip:

```bash
cd nlpia  # make sure you're in the nlpia directory that contains `setup.py`
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


6. Have Fun!

Check out the code examples from the book in `nlpia/nlpia/book/examples` to get ideas:

```bash
cd nlpia/book/examples
ls
```

## Contributing

Help your fellow readers by contributing to your shared code and knowledge.
Here are some ideas for a few features others might find handy.

### Feature 1: Glossary Compiler

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

### Feature 2: Semantic Search

Use a parser to extract only natural language sentences and headings/titles from a list of lines/sentences from an asciidoc book like "Natural Language Processing in Action".
Use a sentence segmenter in https://github.com/totalgood/nlpia/blob/master/src/nlpia/transcoders.py:[nlpia.transcoders] to split a book, like _NLPIA_, into a seequence of sentences.




