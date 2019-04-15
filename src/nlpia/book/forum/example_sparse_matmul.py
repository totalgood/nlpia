"""
>>> import numpy as np
>>> import scipy
>>> id(scipy.dot) == id(np.dot)
True
>>> A = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = scipy.sparse.csr_matrix([[1], [0], [-1]])
>>> A.dot(v)
<3x1 sparse matrix of type '<class 'numpy.int64'>'
    with 3 stored elements in Compressed Sparse Row format>
>>> scipy.dot(A, v)
<3x1 sparse matrix of type '<class 'numpy.int64'>'
    with 3 stored elements in Compressed Sparse Row format>
>>> np.dot(A, v)
<3x1 sparse matrix of type '<class 'numpy.int64'>'
    with 3 stored elements in Compressed Sparse Row format>
"""

import numpy as np
import scipy
id(scipy.dot) == id(np.dot)
# True
A = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
v = scipy.sparse.csr_matrix([[1], [0], [-1]])
A.dot(v)
# <3x1 sparse matrix of type '<class 'numpy.int64'>'
#     with 3 stored elements in Compressed Sparse Row format>
scipy.dot(A, v)
# <3x1 sparse matrix of type '<class 'numpy.int64'>'
#     with 3 stored elements in Compressed Sparse Row format>
np.dot(A, v)
# <3x1 sparse matrix of type '<class 'numpy.int64'>'
#     with 3 stored elements in Compressed Sparse Row format>
