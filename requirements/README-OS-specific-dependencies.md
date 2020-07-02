# OS-Specific Binary Dependencies

## Ubuntu (apt)

On newish Ubuntu OSes you need the python header files:

```bash
$ sudo apt-get install python-dev python3.5-dev python3-virtualenv python-pip build-essential gfortran
```

But python-igraph often requires the binary igraph package
```bash
$ sudo apt-get install python-igraph
```

# SpeechRecognizer requires PyAudio
```bash
$ sudo apt-get install install portaudio19-dev python-pyaudio
```

## Mac OSX (homebrew)

On Mac OSX you need python3 and pip (brew installs the python-dev header files automatically?):
```bash
$ brew install python3
```

SpeechRecognizer requires PyAudio
```bash
$ brew install portaudio
```

Offline SpeechRecognizer requires pocketsphinx and swig
```bash
$ brew install cmu-pocketsphinx
$ brew install swig
```

############################
# scipy requires:
#$ sudo apt-get install gfortran libopenblas-dev liblapack-dev python-scipy python-matplotlib python-numpy

# matplotlib requires:
#$ sudo apt-get install libpng12-dev libfreetype6-dev

# matplotlib requires a backend (see matplotlibrc), tkagg is one option:
#$ sudo apt-get install tcl-dev tk-dev python-tk python3-tk

# pugnlp includes:
# coverage>=4.3.4 nltk>=3.2.2 fuzzywuzzy>=0.15.0 python-slugify>=1.2.4 nlup>=0.5

# optional packages:
# boto==2.46
# editdistance>=0.3.1

