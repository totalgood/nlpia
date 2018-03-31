import os

import matplotlib
matplotlib.use('TkAgg')
import seaborn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pandas as pd

from nlpia.constants import DATA_PATH

np = pd.np


def plot_ply(plyfile=os.path.join(DATA_PATH, 'horse.ply')):
    h = pd.read_csv(os.path.join(DATA_PATH, 'pointcloud.csv.gz'), header=0, index_col=0)
    h = pd.DataFrame(h, columns='x y z'.split())
    h = h.sample(1000).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(h.x, h.y, h.z, c='b', zdir='z', depthshade=True)
    # plt.show()
