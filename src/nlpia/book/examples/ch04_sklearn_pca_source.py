""" Experiments with PCA source code to duplicate results with numpy.svd

>>> from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm
>>> bow_svd, tfidf_svd = lsa_models()  # <1>
>>> prettify_tdm(**bow_svd)
   lion dog nyc cat love apple                                             text
0             1              1                            NYC is the Big Apple.
1             1              1                   NYC is known as the Big Apple.
2             1        1                                            I love NYC!
3             1              1      I wore a hat to the Big Apple party in NYC.
4             1              1                  Come to NYC. See the Big Apple!
5                            1               Manhattan is called the Big Apple.
6                 1                     New York is a big city for a small cat.
7     1           1             The lion, a big cat, is the king of the jungle.
8                 1    1                                     I love my pet cat.
9             1        1                            I love New York City (NYC).
10        1       1                                     Your dog chased my cat.

>>> tdm = bow_svd['tdm'] 
>>> X = tdm.T.values - tdm.T.values.mean(axis=0)  # centers the document-term matrix

>>> ffU, ffS, ffV = _fit_full(self, X, n_components=6)
>>> ffV.round(2)
array([[ 0.16,  0.16, -0.55,  0.59,  0.09, -0.54],
       [-0.12, -0.12,  0.34, -0.16,  0.79, -0.46],
       [-0.71,  0.71,  0.  , -0.  , -0.  , -0.  ],
       [-0.41, -0.41, -0.64, -0.11,  0.34,  0.35],
       [ 0.45,  0.45, -0.39, -0.65,  0.16, -0.02],
       [-0.3 , -0.3 , -0.13, -0.44, -0.49, -0.61]])
>>> svdU, svdS, svdV = np.linalg.svd(X, full_matrices=False)
>>> svdV.round(2)
array([[ 0.16,  0.16, -0.55,  0.59,  0.09, -0.54],
       [ 0.12,  0.12, -0.34,  0.16, -0.79,  0.46],
       [ 0.71, -0.71, -0.  ,  0.  ,  0.  ,  0.  ],
       [-0.41, -0.41, -0.64, -0.11,  0.34,  0.35],
       [ 0.45,  0.45, -0.39, -0.65,  0.16, -0.02],
       [ 0.3 ,  0.3 ,  0.13,  0.44,  0.49,  0.61]])

>>> X = tdm.T.values - tdm.T.values.mean(axis=0)
>>> svdU, svdS, svdV = np.linalg.svd(X)
>>> np.linalg.svd?
>>> np.linalg.svd?
>>> pca_model.transform?
>>> pca_model.transform??
>>> svdU, svdS, svdV = np.linalg.svd(X.T)
>>> np.allclose(svdU, ffV)
False
>>> svdU.round(2)
array([[-0.16,  0.12, -0.71,  0.41, -0.45,  0.3 ],
       [-0.16,  0.12,  0.71,  0.41, -0.45,  0.3 ],
       [ 0.55, -0.34,  0.  ,  0.64,  0.39,  0.13],
       [-0.59,  0.16,  0.  ,  0.11,  0.65,  0.44],
       [-0.09, -0.79, -0.  , -0.34, -0.16,  0.49],
       [ 0.54,  0.46, -0.  , -0.35,  0.02,  0.61]])
>>> ffV.round(2).T
array([[ 0.16, -0.12, -0.71, -0.41,  0.45, -0.3 ],
       [ 0.16, -0.12,  0.71, -0.41,  0.45, -0.3 ],
       [-0.55,  0.34,  0.  , -0.64, -0.39, -0.13],
       [ 0.59, -0.16, -0.  , -0.11, -0.65, -0.44],
       [ 0.09,  0.79, -0.  ,  0.34,  0.16, -0.49],
       [-0.54, -0.46, -0.  ,  0.35, -0.02, -0.61]])

>>> ffV.round(2).T + svdU.round(2)
array([[ 0.  ,  0.  , -1.42,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  1.42,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  , -0.  ,  0.  ,  0.  ,  0.  ],
>>> np.allclose(np.abs(ffV.T), np.abs(svdU))
True
"""
from numpy import linalg
import numpy as np
from six.moves import xrange

class Self:
    pass

self = Self()


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


X = np.array([[-0.09090909, -0.09090909, -0.09090909, -0.09090909, -0.09090909,
        -0.09090909, -0.09090909,  0.90909091, -0.09090909, -0.09090909,
        -0.09090909],
       [-0.09090909, -0.09090909, -0.09090909, -0.09090909, -0.09090909,
        -0.09090909, -0.09090909, -0.09090909, -0.09090909, -0.09090909,
         0.90909091],
       [ 0.45454545,  0.45454545,  0.45454545,  0.45454545,  0.45454545,
        -0.54545455, -0.54545455, -0.54545455, -0.54545455,  0.45454545,
        -0.54545455],
       [-0.36363636, -0.36363636, -0.36363636, -0.36363636, -0.36363636,
        -0.36363636,  0.63636364,  0.63636364,  0.63636364, -0.36363636,
         0.63636364],
       [-0.27272727, -0.27272727,  0.72727273, -0.27272727, -0.27272727,
        -0.27272727, -0.27272727, -0.27272727,  0.72727273,  0.72727273,
        -0.27272727],
       [ 0.54545455,  0.54545455, -0.45454545,  0.54545455,  0.54545455,
         0.54545455, -0.45454545, -0.45454545, -0.45454545, -0.45454545,
        -0.45454545]]).T



def _fit_full(self=self, X=X, n_components=6):
    """Fit the model by computing full SVD on X"""
    n_samples, n_features = X.shape

    # Center data
    self.mean_ = np.mean(X, axis=0)
    print(self.mean_)
    X -= self.mean_
    print(X.round(2))

    U, S, V = linalg.svd(X, full_matrices=False)
    print(V.round(2))
    # flip eigenvectors' sign to enforce deterministic output
    U, V = svd_flip(U, V)

    components_ = V
    print(components_.round(2))

    # Get variance explained by singular values
    explained_variance_ = (S ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var
    singular_values_ = S.copy()  # Store the singular values.

    # Postprocess the number of components required
    if n_components == 'mle':
        n_components = \
            _infer_dimension_(explained_variance_, n_samples, n_features)
    elif 0 < n_components < 1.0:
        # number of components for which the cumulated explained
        # variance percentage is superior to the desired threshold
        ratio_cumsum = stable_cumsum(explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, n_components) + 1

    # Compute noise covariance using Probabilistic PCA model
    # The sigma2 maximum likelihood (cf. eq. 12.46)
    if n_components < min(n_features, n_samples):
        self.noise_variance_ = explained_variance_[n_components:].mean()
    else:
        self.noise_variance_ = 0.

    self.n_samples_, self.n_features_ = n_samples, n_features
    self.components_ = components_[:n_components]
    print(self.components_.round(2))
    self.n_components_ = n_components
    self.explained_variance_ = explained_variance_[:n_components]
    self.explained_variance_ratio_ = \
        explained_variance_ratio_[:n_components]
    self.singular_values_ = singular_values_[:n_components]

    return U, S, V