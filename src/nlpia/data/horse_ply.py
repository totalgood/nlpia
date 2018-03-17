import matplotlib
matplotlib.use('TkAgg')
import seaborn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import *  # noqa

import pandas as pd

from plyfile import PlyData

from nlpia.constants import DATA_PATH

np = pd.np


def plot_ply(plyfile=os.path.join(DATA_PATH, 'horse.ply')):
    horse = PlyData.read(plyfile)
    points = np.array(horse.elements[0].data)
    # facets = np.array(horse.elements[1].data)
    points = np.array([row.tolist() for row in points])
    # pd.DataFrame(points, columns='x y z alpha r g b'.split())
    df = pd.DataFrame(points[:, :3], columns='x y z'.split())
    # df.to_csv('horse.csv')


    h = pd.read_csv('horse.csv', header=None).values[:, :3]
    h = pd.DataFrame(h, columns='x y z'.split())
    h = h.sample(1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(h.x, h.y, h.z, zdir='z', s=20, c=None, depthshade=True)
    plt.show()
