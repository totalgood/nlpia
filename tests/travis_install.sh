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

    # Use the anaconda3 installer
    DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME/.tmp/anaconda3}
    mkdir -p $DOWNLOAD_DIR
    wget http://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh \
        -O $DOWNLOAD_DIR/anaconda3.sh
    chmod +x $DOWNLOAD_DIR/anaconda3.sh && \
        bash $DOWNLOAD_DIR/anaconda3.sh -b -u -p $HOME/anaconda3 && \
        # rm -r -d -f $DOWNLOAD_DIR
    export PATH=$HOME/anaconda3/bin:$PATH
    conda update --yes conda
    conda install pip

    # Configure the conda environment and put it in the path using the provided versions
    if [[ "$ENVIRONMENT_YML" ]]; then
        conda env create -n testenv -f "$ENVIRONMENT_YML"
    else
        conda create -n testenv --yes python=$PYTHON_VERSION pip
    fi
    source activate testenv
    which python
    python --version
elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    echo $DISTRIB
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
