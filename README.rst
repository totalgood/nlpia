NLPIA
=====

The code and data for `Natural Language Processing in
Action <https://www.manning.com/books/natural-language-processing-in-action>`__.

Description
~~~~~~~~~~~

A community-developed book about building Natural Language Processing
pipelines for prosocial chatbots that contribute to communities.

Getting Started
~~~~~~~~~~~~~~~

1. Install `Anaconda <https://docs.anaconda.com/anaconda/install/>`__

2. Clone this repository

.. code:: bash

    # git clone https://github.com/totalgood/nlpia.git
    # cd nlpia

3. Use ``conda-env`` OR ``pip`` to install dependencies

Depending on your OS you may have better luck using conda to install the
dependencies

Use ``conda-env``
^^^^^^^^^^^^^^^^^

The environment.yml file creates a conda environment called
``conda_env_nlpia``

.. code:: bash

    # conda env create -f conda/environment.yml
    # source activate conda_env_nlpia

Use ``pip``
^^^^^^^^^^^

.. code:: bash

    # conda create -y -n conda_env_nlpia
    # source activate conda_env_nlpia
    # conda install -y pip
    # pip install -e .

4. Activate this new environment

.. code:: bash

    # source activate nlpia

5. Install an "editable" ``nlpia`` package in this conda environment
   (also called nlpia)

.. code:: bash

    # pip install -e .

6. Check out the code examples from the book in
   ``nlpia/nlpia/book/examples``

.. code:: bash

    # cd nlpia/book/examples
    # ls
