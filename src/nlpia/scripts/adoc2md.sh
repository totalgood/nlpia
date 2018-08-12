#!/usr/bin/env bash
# Convert asciidoc file to GitHub markdown using asciidoctor and pandoc

# Adapted from: https://gist.github.com/cheungnj/38becf045654119f96c87db829f1be8e#file-script-sh
# Which was adapted from: https://tinyapps.org/blog/nix/201701240700_convert_asciidoc_to_markdown.html

# Requirements
# asciidoctor==1.5.6.1
# pandoc==2.0.0.1
# Install pandoc and asciidoctor

export HOST_OS=$OSTYPE  # linux or darwin17, don't know if it works on Windows

export UNVERSIONED_OS=${HOST_OS::6}
if [ $UNVERSIONED_OS == "darwin" ] ; then
    # since we're not on travis we need to set the BUILD_DIR
    export BUILD_DIR=$HOME/build
    mkdir -p $HOME/build
    export HOST_OS=$UNVERSIONED_OS  # linux or darwin
else
    # we may be on Travis-CI so we need to exit with an error code on any error
    set -e
    export UNVERSIONED_OS="linux"
    export HOST_OS=$UNVERSIONED_OS  # linux or darwin
fi

echo "export HOST_OS=$HOST_OS"


if [ "$HOST_OS" == "linux" ]; then
    sudo apt install asciidoctor
    sudo wget https://github.com/jgm/pandoc/releases/download/2.0.0.1/pandoc-2.0.0.1-1-amd64.deb
    sudo dpkg -i pandoc-2.0.0.1-1-amd64.deb
elif [ "$HOST_OS" == "darwin" ]; then
    brew install asciidoctor pandoc
else
    echo "ERROR: I don't like $HOST_OS. Install a real OS like Linux and try again."
fi

source_file="$1"

# adoc -> docbook
# -b, --backend BACKEND            set output format backend: [html5, xhtml5, docbook5, docbook45, manpage] (default: html5)
asciidoctor -b docbook "$source_file" -o "${source_file}.xml"

# docbook -> md is garbled??
pandoc -f docbook -t 'markdown_github' "${source_file}.xml" -o "${source_file}.md"

# Unicode symbols were mangled Workaround:
iconv -t utf-8 "${source_file}.xml" | pandoc -f docbook -t 'markdown_github' | iconv -f utf-8 > "${source_file}.md"

# Pandoc inserted hard line breaks at 72 characters. Removed like so:
pandoc -f docbook -t 'markdown_github' --wrap=none "${source_file}.xml" -o "${source_file}.md"

# extend line breaks to 120
# pandoc -f docbook -t gfm --columns=120