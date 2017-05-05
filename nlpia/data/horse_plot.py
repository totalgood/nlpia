from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict, 
from builtins import *  # noqa
from past.builtins import basestring   # noqa

from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import pandas as pd
np = pd.np


h = pd.read_csv('pointcloud.csv.gz', header=None).values[:, :3]
h = pd.DataFrame(h, columns='x y z'.split())
h = h.sample(1000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(h.x, h.y, h.z, zdir='z', s=20, c=None, depthshade=True)
plt.show()
