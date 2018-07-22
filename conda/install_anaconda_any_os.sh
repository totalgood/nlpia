#!/usr/bin/env bash

rm -irf ~/Anaconda
OS=MacOSX  # or Linux or Windows
BITS=_64  # or '' for 32-bit
curl https://repo.anaconda.com/archive/ >> tmp.html
FILENAME=$(grep -o -E -e "Anaconda3-[.0-9]+-$OS-x86$BITS\.(sh|exe)" tmp.html | head -n 1)
curl "https://repo.anaconda.com/archive/$FILENAME" > install_anaconda
chmod +x install_anaconda
./install_anaconda -b -p ~/Anaconda
export PATH="$HOME/Anaconda/bin:$PATH"
echo 'export PATH="$HOME/Anaconda/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$HOME/Anaconda/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
rm install_anaconda