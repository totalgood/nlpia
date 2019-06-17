set -e

if [[ -z "$ENVIRONMENT_YML" ]] ; then
    export ENVIRONMENT_YML="conda/environment.yml"
fi
echo "ENVIRONMENT_YML=$ENVIRONMENT_YML"

if [[ -z "$CONDA_ENV_NAME" ]] ; then
    export CONDA_ENV_NAME="venv_nlpia"
fi
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"

# Configure the conda environment and put it in the path using the
# provided versions
# (prefer local venv, since the miniconda folder is cached)
if [[ -f "./$ENVIRONMENT_YML" ]]; then
    if [[ -d "./$CONDA_ENV_NAME" ]] ; then
        conda env update -p "./$CONDA_ENV_NAME" -f "$ENVIRONMENT_YML"
    else
        conda env create -p "./$CONDA_ENV_NAME" -f "$ENVIRONMENT_YML"
    fi
else

if [[ ! -d "./$CONDA_ENV_NAME" ]] ; then
    conda create -p "./$CONDA_ENV_NAME"
    source activate "./$CONDA_ENV_NAME"
    conda install pip
    pip install -r requirements.txt || pip install -r requirements/requirements.txt  || echo "WARN: No requirements.txt found!"
fi

source activate "./$CONDA_ENV_NAME"
