xcode-select --install # Install Command Line Tools if you haven't already.
sudo xcode-select --switch /Library/Developer/CommandLineTools # Enable command line tools

conda install swig

brew tap watsonbox/cmu-sphinx
brew install --HEAD watsonbox/cmu-sphinx/cmu-sphinxbase
brew install --HEAD watsonbox/cmu-sphinx/cmu-pocketsphinx
brew install openal-soft

# install anaconda using graphical installer to put it in /anaconda3/
# cd ~/code
conda env create -n assist -f nlpia/conda/environment.yml python=3.6
conda activate assist
conda install swig
conda install pyaudio
conda install pocketsphinx-python

# brew tap snipsco/homebrew-snips
# brew cask install snips


