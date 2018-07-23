#!/usr/bin/env bash

OS=MacOSX  # or Linux or Windows
BITS=_64  # or '' for 32-bit

if [ $(which conda) == "" ]; then
	rm -irf ~/Anaconda
	curl https://repo.anaconda.com/archive/ > tmp.html
	FILENAME=$(grep -o -E -e "Anaconda3-[.0-9]+-$OS-x86$BITS\.(sh|exe)" tmp.html | head -n 1)
	rm tmp.html
	curl "https://repo.anaconda.com/archive/$FILENAME" > install_anaconda
	chmod +x install_anaconda
	./install_anaconda -b -p ~/Anaconda
	export PATH="$HOME/Anaconda/bin:$PATH"
	echo 'export PATH="$HOME/Anaconda/bin:$PATH"' >> ~/.bashrc
	echo 'export PATH="$HOME/Anaconda/bin:$PATH"' >> ~/.bash_profile
	source ~/.bash_profile
	rm install_anaconda
fi

mkdir -p ~/code
cd ~/code
git clone https://github.com/totalgood/nlpia
cd ~/code/nlpia
conda install -y pip  # <1>
pip install --upgrade pip  # <2>
conda env create -n nlpiaenv -f conda/environment.yml  # <3>
source activate nlpiaenv  # <4>
pip install --upgrade pip  # <5>
pip install -e .  # <6>