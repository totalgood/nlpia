#!/usr/bin/bash

# bootstrap an NLPIA-compatible environment using anaconda environments

if [ $# -eq 1 ] ; then
    export HOST_OS=$1  # linux or darwin
else
	export HOST_OS=$OSTYPE  # linux or darwin17
fi

export UNVERSIONED_OS=${HOST_OS::6}
if [ $UNVERSIONED_OS == "darwin" ] ; then
    # since we're not on travis we need to set the BUILD_DIR
    export BUILD_DIR=$HOME/build
    mkdir -p $HOME/build
    export HOST_OS=$UNVERSIONED_OS  # linux or darwin
else
    # we're on travis so we need to exit with an error code on any error
    set -e
    export UNVERSIONED_OS="linux"
	export HOST_OS=$UNVERSIONED_OS  # linux or darwin
fi

echo "export HOST_OS=$HOST_OS"

set -e

# Use the anaconda3 installer
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME/.tmp/anaconda3}
mkdir -p $DOWNLOAD_DIR

if [ "$HOST_OS" == "linux" ] ; then
    wget http://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O $DOWNLOAD_DIR/anaconda3.sh
    chmod +x $DOWNLOAD_DIR/anaconda3.sh
    bash $DOWNLOAD_DIR/anaconda3.sh -b -u -p $HOME/anaconda3
elif [ "$HOST_OS" -eq "darwin" ] ; then
    wget https://repo.continuum.io/archive/Anaconda2-5.1.0-MacOSX-x86_64.sh -O $DOWNLOAD_DIR/anaconda3.sh
    chmod +x $DOWNLOAD_DIR/anaconda3.sh
    bash $DOWNLOAD_DIR/anaconda3.sh -b -u -p $HOME/anaconda3
else  # WARNING Windows (almost certainly will not work!!!!)
	echo "Windoze enviornment detected with OSTYPE==$OSTYPE"
	echo "WARNING: This almost certainly will not work!!!!"
	wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Windows-x86_64.exe -O $DOWNLOAD_DIR/anaconda3.exe
    chmod +x $DOWNLOAD_DIR/anaconda3.exe
    $DOWNLOAD_DIR/anaconda3.exe -b -u -p $HOME/anaconda3
fi
# chmod +x $DOWNLOAD_DIR/anaconda3.sh
# bash $DOWNLOAD_DIR/anaconda3.sh -b -u -p $HOME/anaconda3
 # rm -r -d -f $DOWNLOAD_DIR
export PATH=$HOME/anaconda3/bin:$PATH
conda update --yes conda
conda install --yes pip
