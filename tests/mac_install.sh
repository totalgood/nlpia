# Only the conda install worked on Mac but it won't work on Windows:
conda install -c conda-forge python-anno


############################################
# In the end all of this FAILED!!!!
#
# even this failed when it pointed directly to the header files that gcc couldn't find:
pip install pip install --global-option=build_ext --global-option="-I/Users/hobsonlane//anaconda3/include/" annoy

xcode-select --install
xcode-select --reset
echo "IMPORTANT: ACCEPT THE LICENSE AGREEMENT IN A SEPARATE GUI WINDOW!!!!!!!!"

# uninstall brew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"

xcode-select --install
xcode-select --reset
echo "IMPORTANT: ACCEPT THE LICENSE AGREEMENT IN A SEPARATE GUI WINDOW!!!!!!!!"

# install brew:
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
xargs brew install < ~/code/brew_leaves.txt
brew install python
# install py2.7 annoy in brew env
/usr/local/bin/pip install annoy
# install py3 annoy in brew env
/usr/local/bin/pip3 install annoy

# install anaconda3 (python3.7)
conda create -n rnet
conda activate rnet
conda install pip
pip install --upgrade pip
#       Successfully uninstalled pip-10.0.1
# Successfully installed pip-18.1
pip install annoy
# Collecting annoy
# Installing collected packages: annoy
# Successfully installed annoy-1.14.0
# so then pip install annoy works in the conda env because it reuses root install?
/usr/local/bin/pip install annoy # works fine using brew-installed python brew
/usr/local/bin/pip3 install annoy

## All the environment info
brew info gcc
# gcc: stable 8.2.0 (bottled), HEAD
# Not installed
which -a pip
# /Users/hobsonlane/anaconda3/bin/pip
# /usr/local/bin/pip
which -a pip3
# /Users/hobsonlane/anaconda3/bin/pip3
# /usr/local/bin/pip3
which -a python
# /Users/hobsonlane/anaconda3/bin/python
# /usr/local/bin/python
# /usr/bin/python
conda env list
# base                     /Users/hobsonlane/anaconda3
# rnet                  *  /Users/hobsonlane/anaconda3/envs/rnet
python -c 'import annoy; print(annoy.__file__)'
# /Users/hobsonlane/anaconda3/lib/python3.7/site-packages/annoy/__init__.py
pip install annoy
# Requirement already satisfied: annoy in ./anaconda3/lib/python3.7/site-packages (1.14.0)
pip --version
# pip 18.1 from /Users/hobsonlane/anaconda3/lib/python3.7/site-packages/pip (python 3.7)
# SHELL=/bin/bash
# CONDA_SHLVL=2
# CONDA_PROMPT_MODIFIER=
# CONDA_EXE=/Users/hobsonlane/anaconda3/bin/conda
# CONDA_PREFIX_1=/Users/hobsonlane/anaconda3
# CONDA_PREFIX=/Users/hobsonlane/anaconda3/envs/rnet
# PATH=/Users/hobsonlane/anaconda3/envs/rnet/bin:/Users/hobsonlane/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Applications/Wireshark.app/Contents/MacOS


# So it looks like you just need to install the site-packages version of annoy (in brew's pip) and it will be reused by all conda envs

conda env remove -n rnet  # no tensorflow for python3.7
conda create -n rnet python==3.6
