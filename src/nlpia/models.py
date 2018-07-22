#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

from pandas import np


class LinearRegressor(object):

    def fit(self, X, y):
        """ Compute average slope and intercept for all X, y pairs

        Arguments:
          X (np.array): model input (independent variable)
          y (np.array): model output (dependent variable)

        Returns:
          Linear Regression instance with `slope` and `intercept` attributes

        References:
          Based on: https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

        >>> n_samples = 100
        >>> X = np.arange(100).reshape((n_samples, 1))
        >>> slope, intercept = 3.14159, -4.242
        >>> y = 3.14 * X + np.random.randn(*X.shape) + intercept
        >>> line = LinearRegressor()
        >>> line.fit(X, y)
        <nlpia.models.LinearRegressor object ...
        >>> abs(line.slope - slope) < abs(0.02 * (slope + 1))
        True
        >>> abs(line.intercept - intercept) < 0.2 * (abs(intercept) + 1)
        True
        """

        # initial sums
        n = float(len(X))
        sum_x = X.sum()
        sum_y = y.sum()
        sum_xy = (X * y).sum()
        sum_xx = (X**2).sum()

        # formula for w0
        self.slope = (sum_xy - (sum_x * sum_y) / n) / (sum_xx - (sum_x * sum_x) / n)

        # formula for w1
        self.intercept = sum_y / n - self.slope * (sum_x / n)

        return self

    def predict(self, X):
        """ self.slope * X + self.intercept """
        return self.slope * X + self.intercept


class OneNeuronRegressor(object):
    """ Full batch learning using the Delta rule (weights += error * weights)

    >>> n_samples = 100
    >>> x = np.random.randn(n_samples, 1)  # 100 random x values
    >>> slope = 3.1
    >>> intercept = -2.7
    >>> noise = .1
    >>> noise = noise * np.random.randn(*x.shape)
    >>> y = slope * x + intercept + noise
    >>> n_epochs = 10000
    >>> nn = OneNeuronRegressor(alpha=0.01, n_iter=1)
    >>> error = np.zeros(n_epochs)
    >>> for i in range(n_epochs):
    ...     error[i] = np.abs(nn.delta(x, y)).sum()
    ...     nn = nn.fit(x, y)
    ...     # print(nn.W, error[i])
    >>> nn.W.round(1)  # intercept, slope
    array([[-2.7,  3.1]])

    X = pca_topic_vectors[['topic4']].values[:5, :]
    y = scores['compound'].reshape(len(scores), 1).values[:5, :]

    nn = OneNeuronRegressor(n_iter=1)
    for i in range(3):
        print('-' * 10)
        print(nn.W1)
        print(nn.predict(X))
        print(nn.error(X, y))
        print(nn.delta(X, y))
        nn = nn.fit(X, y, 1)
        print(nn.W1)
        print(pd.DataFrame(nn.error(X, y)).mean())[0]
    """

    def __init__(self, n_inputs=1, n_iter=1000, alpha=0.01):
        self.n_inputs = n_inputs
        self.n_outputs = 1
        self.W = np.random.randn(self.n_outputs, self.n_inputs + 1)
        self.n_iter = n_iter
        self.alpha = alpha

    def delta(self, X, y):
        X = getattr(X, 'values', X).reshape(len(X), 1)
        return (y.reshape((len(X), 1)) - self.predict(X)).reshape((len(X),))

    def homogenize(self, X):
        X = getattr(X, 'values', X).reshape(len(X), 1)
        X_1 = np.ones((len(X), self.n_inputs + 1))
        X_1[:, 1:] = getattr(X, 'values', X)
        return X_1

    def fit(self, X, y, n_iter=None):
        """w = w + α * δ * X"""
        self.n_iter = self.n_iter if n_iter is None else n_iter
        X = getattr(X, 'values', X).reshape(len(X), 1)
        X_1 = self.homogenize(X)
        for i in range(self.n_iter):
            for i in range(0, len(X), 10):  # minibatch learning for numerical stability
                batch = slice(i, min(i + 10, len(X)))
                Xbatch, ybatch = X[batch, :], y[batch]
                X_1_batch = X_1[batch, :]
                self.W += (self.alpha / len(X) ** 1.5) * (
                    self.delta(Xbatch, ybatch).reshape((len(Xbatch), 1)).T.dot(X_1_batch))
        return self

    def predict(self, X):
        X_1 = self.homogenize(X)
        return self.W.dot(X_1.T).T
