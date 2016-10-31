#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from nlp_in_action.skeleton import fib

__author__ = "Hobson Lane"
__copyright__ = "Hobson Lane"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
