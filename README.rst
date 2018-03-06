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

3. Create a conda environment called nlpia and install its requirements

.. code:: bash

    # cd nlpia
    # conda env create -f conda/environment.yml

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
