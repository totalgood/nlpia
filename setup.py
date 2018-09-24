#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for nlpia.

    This file was generated with PyScaffold 3.0.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: http://pyscaffold.readthedocs.org/
"""

import sys
from setuptools import setup

# Add here console scripts and other entry points in ini-style format
# audio = nlpia.scripts.audio:main
entry_points = ""
# [console_scripts]
# renderbook = nlpia.scripts.countpages:main
# cleandialog = nlpia.scripts.cleandialog:main
# downloadgdrive = nlpia.scripts.google_drive:main
# """

pyscaffold_ver = '3.1'
if sys.version_info[0] < 3:
    pyscaffold_ver = '2.5.10'


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['pyscaffold=={}'.format(pyscaffold_ver)] + sphinx,
          entry_points=entry_points,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
