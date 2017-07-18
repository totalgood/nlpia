#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pandas as pd
from pandas import np


class LinearRegressor:

    def fit(self, X, y):
        """ Compute average slope and intercept for all X, y pairs

        Arguments:
          X (np.array): model input (independent variable)
          y (np.array): model output (dependent variable)

        Returns:
          Linear Regression instance with `slope` and `intercept` attributes

        References:
          Based on: https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb
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


class OneNeuronRegressor:
    """ FIXME: DOES NOT CONVERGE TO THE SAME ANSWER AS SGDREGRESSOR!

    X = np.range(10).reshape(None, 1)
    y = 3.14 * X + np.random.randn(X.shape)

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

    def __init__(self, n_inputs=1, n_iter=1000, alpha=0.1):
        self.n_inputs = n_inputs
        self.n_outputs = 1
        self.W = np.random.randn(self.n_outputs, self.n_inputs + 1)
        self.n_iter = n_iter
        self.alpha = alpha

    def delta(self, X, y):
        return (y.reshape((len(X), 1)) - self.predict(X.reshape(len(X), 1))).reshape((len(X),))

    def homogenize(self, X):
        X_1 = np.ones((len(X), self.n_inputs + 1))
        X_1[:, 1:] = X
        return X_1

    def fit(self, X, y, n_iter=None):
        """w = w + α * δ * X"""
        self.n_iter = self.n_iter if n_iter is None else n_iter
        X_1 = self.homogenize(X)
        for i in range(self.n_iter):
            self.W += self.alpha * self.delta(X, y).reshape((len(X), 1)).T.dot(X_1)
        return self

    def predict(self, X):
        X_1 = self.homogenize(X)
        return self.W.dot(X_1.T).T
