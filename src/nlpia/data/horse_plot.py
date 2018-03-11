# import matplotlib
# matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pandas as pd
np = pd.np


h = pd.read_csv('pointcloud.csv.gz', header=0, index_col=0)
h = pd.DataFrame(h, columns='x y z'.split())
h = h.sample(1000).copy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(h.x, h.y, h.z, c='b', zdir='z', depthshade=True)
plt.show()
