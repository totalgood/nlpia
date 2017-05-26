# -*- coding: utf-8 -*-
""" Unrestricted Boltzman Machine Example from Lecture 11, slide 35:

>>> q11_4 = BoltzmanMachine(bv=[0., 0.], bh=[0., 0], Whh=[[0, -1],[-1, 0]], Wvv=np.zeros((2, 2)), Wvh=[[2., 0], [0., 1.]])
>>> q11_4.configurations()
    v1  v2  h1  h2   -E   exp(-E)    p(v,h)      p(v)
0    0   0   0   0  0.0  1.000000  0.025195  0.084855
1    0   0   0   1  0.0  1.000000  0.025195  0.084855
2    0   0   1   0  0.0  1.000000  0.025195  0.084855
3    0   0   1   1 -1.0  0.367879  0.009269  0.084855
4    0   1   0   0  0.0  1.000000  0.025195  0.144074
5    0   1   0   1  1.0  2.718282  0.068488  0.144074
6    0   1   1   0  0.0  1.000000  0.025195  0.144074
7    0   1   1   1  0.0  1.000000  0.025195  0.144074
8    1   0   0   0  0.0  1.000000  0.025195  0.305048
9    1   0   0   1  0.0  1.000000  0.025195  0.305048
10   1   0   1   0  2.0  7.389056  0.186170  0.305048
11   1   0   1   1  1.0  2.718282  0.068488  0.305048
12   1   1   0   0  0.0  1.000000  0.025195  0.466023
13   1   1   0   1  1.0  2.718282  0.068488  0.466023
14   1   1   1   0  2.0  7.389056  0.186170  0.466023
15   1   1   1   1  2.0  7.389056  0.186170  0.466023


"""
from __future__ import print_function, division, unicode_literals, absolute_import

import numpy as np
import pandas as pd
from itertools import product
assert(product)


def listify(x):
    """ Coerce iterable object into a list or turn a scalar into single element list

    >>> listify(1.2)
    [1.2]
    >>> listify(range(3))
    [0, 1, 2]
    """
    try:
        return list(x)
    except:
        if x is None:
            return []
        return [x]


def tablify(*args):
    r"""
    >>> tablify(range(3), range(10, 13))
    [[0, 10], [1, 11], [2, 12]]
    """
    table = []
    args = [listify(arg) for arg in args]
    for row in zip(*args):
        r = []
        for x in row:
            r += listify(x)
        table += [r]
    return table
    return [sum([listify(el) for el in row]) for row in zip(*args)]


def framify(*args, **kwargs):
    # print(args)
    table = tablify(*args)
    # print(table)
    columns = kwargs.get('columns', range(len(table[0])))
    return pd.DataFrame(table, columns=range(len(args)) if columns is None else columns)


def make_wide(x):
    x = framify(x).values
    if x.shape[0] > x.shape[1]:
        return x.T
    return x


def make_tall(x):
    x = framify(x).values
    if x.shape[0] < x.shape[1]:
        return x.T
    return x


class BoltzmanMachine(object):

    def __init__(self, Wvh, Wvv=None, Whh=None, bh=None, bv=None, T=0):
        self.T = T
        self.Nv, self.Nh = framify(Wvh).shape
        self.bh = np.zeros(self.Nh) if bh is None else bh
        self.bv = np.zeros(self.Nv) if bv is None else bv
        self.Wvv = np.zeros(self.Nv, self.Nn) if Wvv is None else framify(Wvv).values
        self.Whh = np.zeros(self.Nh, self.Nh) if Whh is None else framify(Whh).values
        self.Wvh = np.zeros(self.Nv, self.Nh) if Wvh is None else framify(Wvh).values

    def energy(self, v, h=None):
        """Compute the global energy for the current joint state of all nodes

        >>> q11_4 = BoltzmanMachine(bv=[0., 0.], bh=[-2.], Whh=np.zeros((1, 1)), Wvv=np.zeros((2, 2)), Wvh=[[3.], [-1.]])
        >>> q11_4.configurations()

        >>> v1v2h = product([0, 1], [0, 1], [0, 1])
        >>> E = np.array([q11_4.energy(v=x[0:2], h=[x[2]]) for x in v1v2h])
        >>> expnegE = np.exp(-E)
        >>> sumexpnegE = np.sum(expnegE)
        >>> pvh = np.array([ene / sumexpnegE for ene in expnegE])
        >>> pv = [0] * len(df)
        >>> num_hid_states = 2 ** self.Nh
        >>> for i in range(len(df)):
                j = int(i / num_hid_states)
                pv[i] = sum(pvh[k] for k in range(j * num_hid_states, (j + 1) * num_hid_states))
        >>> pd.DataFrame(tablify(v1v2h, -E, expnegE, pvh, pv), columns='v1 v2 h -E exp(-E) p(v,h), p(v)'.split())
        """
        h = np.zeros(self.Nh) if h is None else h
        negE = np.dot(v, self.bv)
        negE += np.dot(h, self.bh)
        for j in range(self.Nv):
            for i in range(j):
                negE += v[i] * v[j] * self.Wvv[i][j]
        for i in range(self.Nv):
            for k in range(self.Nh):
                negE += v[i] * h[k] * self.Wvh[i][k]
        for l in range(self.Nh):
            for k in range(l):
                negE += h[k] * h[l] * self.Whh[k][l]
        return -negE

    def configurations(self):
        vh = [[0, 1]] * self.Nv + [[0, 1]] * self.Nh
        vh = np.array(list(product(*vh)))
        # print(vh)
        vcols = ['v' + str(i + 1) for i in range(self.Nv)]
        hcols = ['h' + str(i + 1) for i in range(self.Nh)]
        columns = vcols + hcols + '-E exp(-E) p(v,h)'.split()
        E = np.array([self.energy(v=x[0:self.Nv], h=x[self.Nv:self.Nv + self.Nh]) for x in vh])
        expnegE = np.exp(-E)
        sumexpnegE = np.sum(expnegE)
        pvh = np.array([ene / sumexpnegE for ene in expnegE])
        df = framify(vh, -E, np.exp(-E), pvh, columns=columns)
        pv = [0] * len(df)
        # number of possible hidden unit states to sum Pvh over to get Pv
        num_hid_states = 2 ** self.Nh
        for i in range(len(df)):
            j = int(i / num_hid_states)
            pv[i] = sum(df['p(v,h)'][k] for k in range(j * num_hid_states, (j + 1) * num_hid_states))
        df['p(v)'] = pv
        return df


class Hopfield(object):
    r""" Hopfield network simulation/training

    Q11.5: Wac=Wbc=1, Wce=Wcd=2, Wbe=−3, Wad=−2, and Wde=3
    What are the lowest and second-lowest energy configurations?

    >>> W = np.zeros((5, 5))
    >>> W[0,2] = W[1,2] = 1
    >>> W[2,4] = W[2,3] = 2
    >>> W[1,4] = -3
    >>> W[0,3] = -2
    >>> W[3,4] = 3
    >>> b = np.zeros(5)
    >>> h = Hopfield(W, b, verbosity=1)
    >>> h.activate()
    >>> h.activate()
    """

    def __init__(self, W, b=None, initial_state=None, random_seed=None, verbosity=0):
        self.random_seed = random_seed
        self.verbosity = verbosity
        self.N = W.shape[0]
        self.b = np.zeros(self.N) if b is None else np.array(b)
        self.W = np.array(W)
        if np.all(self.W.T + self.W == (self.W.T + self.W).T):
            self.W += self.W.T
        self.reset(initial_state)

    def reset(self, initial_state=None):
        np.random.seed(self.random_seed)
        self.E = None
        self.state = np.random.binomial(1, .5, self.N) if initial_state is None else np.array(initial_state)
        self.high_energies = np.array([-np.inf] * 11)
        self.low_energies = np.array([np.inf] * 11)

    def simulate(self, num_epochs=100):
        for i in range(num_epochs):
            self.activate()

    def activate(self, sequence=None):
        if sequence is None:
            sequence = np.arange(self.N)
            np.random.shuffle(sequence)
        for node in sequence:
            s = self.state[node]
            E = self.energy()
            self.state[node] = int(np.dot(self.W[node], self.state) > 0)
            self.energy()
            if self.verbosity:
                print('{: 3d}: {:1d} ({}) => {:1d} ({})'.format(
                    node, s, round(E, 5), self.state[node], round(self.E, 5)))
        return self.state

    def energy(self):
        r""" Compute the global energy for the current joint state of all nodes

        - sum(s[i] * b[i]) - sum([s[i]*s[j]*W[i,j] for (i, j) in product(range(N), range(N)) if i<j)])

        E = − ∑ s i b i − ∑
        i i< j
        s i s j w ij
        """
        s, b, W, N = self.state, self.b, self.W, self.N
        self.E = - sum(s * b) - sum([s[i] * s[j] * W[i, j] for (i, j) in product(range(N), range(N)) if i < j])
        self.low_energies[-1] = self.E
        self.low_energies.sort()
        self.high_energies[-1] = self.E
        self.high_energies.sort()
        self.high_energies = self.high_energies[::-1]
        return self.E
        # for i in sequence:
        #     for w in self.W[i]:
        #         if w:
