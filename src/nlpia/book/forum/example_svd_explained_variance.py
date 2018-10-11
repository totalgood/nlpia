"""
>>> import numpy as np
>>> import sklearn.decomposition
>>> svd = sklearn.decomposition.TruncatedSVD(2)
>>> A = np.random.randn(1000,100)
>>> svd.fit(A)
TruncatedSVD(algorithm='randomized', n_components=2, n_iter=5,
       random_state=None, tol=0.0)
>>> A_2D = svd.transform(A)
>>> np.var(A_2D, axis=0)
array([1.7039018 , 1.65362273])
>>> var = A.shape[1] * np.var(A_2D, axis=0) / np.var(A, axis=0).sum()
>>> var
array([1.6955373 , 1.64550506])
>>> np.abs(var - svd.explained_variance_).round(2)
array([0., 0.])
"""
