# NLPIA

The community - driven code for [**N**atural **L**anguage **P**rocessing **i**n **A**ction](https://bit.ly/nlpiabook).

# Description

A community-developed book about building socially responsible NLP pipelines that give back to the communities they interact with.

# Getting Started

You'll need a bash terminal on your machine. 
[Git-bash](https://git-scm.com/downloads) by GitHub has installers for all three major OSes. 

Once you have Git installed, launch a bash terminal. 
It will usually be found among your other applications with the name `git-bash`. 

1. Install Python

[Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://repo.continuum.io/miniconda/) are good ways to install python3 without messing up your system python on any machine (including Windows and Mac OSX). 
Another advantage of the `conda` package manager that miniconda installs is that it can install some packages that pip cannot. 
There are often binary dependencies for python packages like the `QtPython` dependency for `matplotlib` or `pyaudio` for `SpeechRecognition`.
 
```bash
export OS=Linux  # or MacOSX or Windows
if [ "$OS" == "Windows"; then
    curl https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe -O
    ./Anaconda3-5.2.0-Windows-x86_64.exe /S /D=$HOME/miniconda3/  # this will launch a GUI despite the "/S ilent" arg
else
    wget -c "https://repo.continuum.io/miniconda/Miniconda3-latest-${OS}-x86_64.sh" -o $HOME/install_miniconda3.sh
    chmod +x $HOME/install_miniconda3.sh
    cd $HOME
    ./install_miniconda.sh -b -p $HOME/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    echo 'export PATH=$HOME/miniconda/bin:$PATH' >> $HOME/.bash_profile
    bash $HOME/.bash_profile  # or just restart your terminal
fi
```

If you're installing Anaconda using the Windows GUI be sure to check the box to install it in your PATH variable. 
That way you can install pip in the next step:

2. Clone this repository

```bash
git clone https://github.com/totalgood/nlpia.git
cd nlpia
```

3. Install `nlpia` and its dependencies

In most cases, conda will be able to install python packages faster and more reliably than pip. 

### Use `conda-env`


Create a conda environment called `nlpiaenv`

```bash
conda env create -n nlpiaenv -f conda/environment.yml
conda install pip  # to get the latest version of pip
```

Whenever you want to be able to import or run any `nlpia` modules, you'll need to activate this conda environment first:

```bash
source activate nlpiaenv
```

Skip to Step 4 if you successfully created and activated an environment containing the `nlpia` packages.

### Use `pip`

Linux-based OSes like Ubuntu and OSX come with C++ compilers built-in, so you may be able to install the dependencies using pip instead of `conda`. 
But if you're on Windows and you want to install packages, like `python-levenshtein` that need compiled C++ libraries, you'll need a compiler. 
Fortunately Microsoft still lets you [download a compiler for free](https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29), just make sure you follow the links to the Visual Studio "Build Tools" and not the entire Visual Studio package.

If you have a compiler on your OS you may be able to install `nlpia` using pip, without using `conda`:

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
source activate nlpia
```

5. Install an "editable" `nlpia` package in this conda environment (called `nlpiaenv`)

```bash
pip install -e .
```

6. Check out the code examples from the book in `nlpia/nlpia/book/examples`

```bash
cd nlpia/book/examples
ls
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
Use a sentence segmenter in https://github.com/totalgood/nlpia/blob/master/src/nlpia/transcoders.py:[nlpia.transcoders] to split a book, like _NLPIA_, into a seequence of sentences.




