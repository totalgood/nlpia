# NLPIA

The code and data for [Natural Language Processing in Action](https://www.manning.com/books/natural-language-processing-in-action).

## Description

A community-developed book about building Natural Language Processing pipelines for prosocial chatbots that contribute to communities.

## Getting Started

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)  

2. Clone this repository

```bash
# git clone https://github.com/totalgood/nlpia.git
# cd nlpia
```

3. Use `conda-env` OR `pip` to install dependencies

Depending on your OS you may have better luck using conda to install the dependencies

#### Use `conda-env`

The environment.yml file creates a conda environment called `conda_env_nlpia`

```bash
# conda env create -f conda/environment.yml
# source activate conda_env_nlpia
```

#### Use `pip`

```bash
# conda create -y -n conda_env_nlpia
# source activate conda_env_nlpia
# conda install -y pip
# pip install -e .
# pip install -r requirements-deep.txt
```

The chatbots (including TTS and STT audio drivers) that come with `nlpia` may not be compatible with Windows due to problems install `pycrypto`.  
If you are on a Linux or Darwin (Mac OSX) system or want to try to help us debug the pycrypto problem feel free to install the chatbot requirements:

```bash
# pip install -r requirements-chat.txt
# pip install -r requirements-voice.txt
```

4. Activate this new environment

```bash
# source activate nlpia
```

5. Install an "editable" `nlpia` package in this conda environment (also called nlpia)

```bash
# pip install -e .
```

6. Check out the code examples from the book in `nlpia/nlpia/book/examples`

```bash
# cd nlpia/book/examples
# ls
```
