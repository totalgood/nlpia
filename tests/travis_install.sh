#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is taken from Scikit-Learn (http://scikit-learn.org/)
#
# THIS SCRIPT IS SUPPOSED TO BE AN EXAMPLE. MODIFY IT ACCORDING TO YOUR NEEDS!

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    if [[ -f "$HOME/miniconda/bin/conda" ]]; then
        echo "Skip install conda [cached]"
    else
        # By default, travis caching mechanism creates an empty dir in the
        # beginning of the build, but conda installer aborts if it finds an
        # existing folder, so let's just remove it:
        rm -rf "$HOME/miniconda"

    # Use the anaconda3 installer
    DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME/.tmp/anaconda3}
    mkdir -p $DOWNLOAD_DIR
    wget -q http://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O $DOWNLOAD_DIR/anaconda3.sh
    chmod +x $DOWNLOAD_DIR/anaconda3.sh && bash $DOWNLOAD_DIR/anaconda3.sh -b -u -p $HOME/anaconda3
    # rm -r -d -f $DOWNLOAD_DIR
    # export PATH=$HOME/anaconda3/bin:$PATH
    conda update -q -y conda
    conda install -q -y pip
    conda install -q -y swig
    echo "creating conda environemnt -n testenv -f $ENVIRONMENT_YML"
    # Configure the conda environment and put it in the path using the provided versions
    if [[ -f "$ENVIRONMENT_YML" ]]; then
        conda env create -n testenv -f "$ENVIRONMENT_YML"
    else
        echo "WARNING: Unable to find an environment.yml file !!!!!!"
        conda create -q -n testenv --yes python=$PYTHON_VERSION pip
    fi
    source activate testenv
    echo "Installing pip with conda quietly"
    conda install -q -y pip

    # download spacy English language model
    echo "Installing pip with conda quietly"
    conda install spacy
    echo "Downloading spacy language model"
    python -m spacy download en

    # download NLTK punkt, Penn Treebank, and wordnet corpora
    python -c "import nltk; nltk.download('punkt'); nltk.download('treebank'); nltk.download('wordnet');"
    which python
    python --version

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    echo $DISTRIB
    apt-get install -y build-essential swig gfortran
    apt-get install -y python-dev python3-dev python-pip python3-pip

    apt-get install -y python-igraph

    # SpeechRecognizer requires PyAudio
    apt-get install -y portaudio19-dev python-pyaudio python3-pyaudio

    # for scipy
    apt-get install -y libopenblas-dev liblapack-dev
    apt-get install -y python-scipy python3-scipy

    # for matplotlib:
    apt-get install -y libpng12-dev libfreetype6-dev
    apt-get install -y tcl-dev tk-dev python-tk python3-tk
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
