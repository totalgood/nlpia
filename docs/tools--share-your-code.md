# Share Your Code

Python is all about making it easy to share your work.

To share your code package (a folder full of *.py files) and make it easy for others to use it you need to build a package (add a setup.py file) and register it with package managers like `pip` that use `pypi` (the Python.org "Cheese Shop") or `conda` which uses `binstar` at anaconda.org. That way your users can `pip install YOUR_PACKAGE` or `conda install YOUR_PACKAGE`

## Build a PyPi Source Code Package

The key is to use the [PyScaffold](https://pypi.python.org/pypi/PyScaffold) command line tool `putup` to set up your python project with a best-practice `setup.py` and `setup.cfg` files, as well as a standard directory structure.

I like to run this from within a user dir where I keep all my source code, and I like to include

```bash
pip install PyScaffold pre-commit django django_extensions
mkdir -p ~/src/
cd ~/src/
putup --with-travis --with-tox --with-precommit --with-django myprojectname
```

### `setup.py`

Check out the instructions in the comments within the [`setup.cfg`](https://pyscaffold.readthedocs.io/en/v2.5.7/configuration.html) file (which is used by `setup.py`). This file was created during PyScaffold `putup`. The comments show you where to put in your version number and description categories, all without touching the `setup.py` file directly.

Test you customized `setup.py` with

```bash
pip install -e /path/to/project/dir/ --upgrade
```

If you don't have pip installed, or prefer going old-school you can do:

```bash
cd /path/to/project/dir/
python setup.py install
```

### Upload to PyPI

Now that `setup.cfg` is customized and `python setup.py install` works for you locally, you can register and push your package to pypi using Peter Downs' [instructions](http://peterdowns.com/posts/first-time-with-pypi.html). Below is an abbreviated set of commands that should work. Build the package by running the following command within your project folder (along side the `setup.py` file):

```bash
cd /path/to/project/dir/
python setup.py sdist
```

To upload it to the cheese shop you'll need [register an account on pypi](https://pypi.python.org/pypi?%3Aaction=register_form) and then you'll need to put your account information into a `~.pypirc` file so you don't have to type your credentials all the time. Here's what mine typically look like:

```bash
[distutils]
index-servers =
    pypi
    pypitest
[pypirc]
servers = 
    pypi
    pypitest
[server-login]
username:your_username
password:your_password
[pypi]
repository:https://pypi.python.org/pypi
username:your_username
password:your_password
[pypitest]
repository:https://testpypi.python.org/pypi
username:your_username
password:your_password
```

This will allow you to upload your package to `pypitest` to make sure everything looks OK (mainly the README.rst):

```bash
python setup.py register -r pypitest
python setup.py sdist upload -r pypitest
```

To [upload](https://docs.python.org/3.1/distutils/uploading.html) your source distribution package to the real `pypi` "Cheese Shop" just run:

```bash
python setup.py register -r pypi
python setup.py sdist upload -r pypi
```

## Build a Conda Binary Package

Once the `pypi` package is uploaded, building and uploading a conda package is even easier. All you need to do is:

First you need to [signup for account at Anaconda.org](https://anaconda.org/#sign-up).

Next you'll need to put your new anaconda.org user name in a `.condarc` file so `conda` knows where to put your packages. Here's an example:

```yaml
# filename: ~/.condarc
channels:
  - defaults
  - http://conda.binstar.org/YOUR_ANACONDA_SITE_USERNAME
```

Because your `pypi` source code is already registered a pypi, you can just do this to rebuild it as a binary package and upload it to anaconda.org (binstar):

conda skeleton pypi YOUR_PACKAGE_NAME
conda build YOUR_PACKAGE_NAME


Now you can move on over to a new conda environment (or your friend's computer) and try to install it to make sure everything goes OK.

conda install YOUR_PACKAGE_NAME