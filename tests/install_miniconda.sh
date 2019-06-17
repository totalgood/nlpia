set -e

if [[ -z "$PYTHON_VERSION" ]] ; then
    export PYTHON_VERSION=3.6
fi
echo "PYTHON_VERSION=$CONDA_PYTHON_VERSION"

if [[ -z "$CONDA_DIR" ]] ; then
    export CONDA_DIR="$HOME/miniconda"
fi
echo "CONDA_DIR=$CONDA_DIR"

if [[ -z "$CONDA_ENV_NAME" ]] ; then
    export CONDA_ENV_NAME="venv_nlpia"
fi
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"


if [[ -f "$CONDA_DIR/bin/conda" ]]; then
    echo "Skip install conda (it has already been installed and cached)"
else
    # By default, travis caching mechanism creates an empty dir in the
    # beginning of the build, but conda installer aborts if it finds an
    # existing folder, so let's just remove it:
    rm -rf "$CONDA_DIR"
    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p "$CONDA_DIR"
fi
export PATH=$CONDA_DIR/bin:$PATH
# Make sure to use the most updated version
conda update --yes conda
conda install pip


