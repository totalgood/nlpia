# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.graph_objs as go

from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import plotly.plotly as plotly

import pandas as pd

np = pd.np


def scatter_3d(df, labels=None, depthshade=True):
    df = getattr(df, 'embedding_', df)
    labels = df[labels] if (isinstance(labels, *(int, str, bytes)) and
                            labels in getattr(df, 'columns', set())) else labels
    labels = np.array(np.zeros(shape=(len(df),)) if labels is None else labels)
    if isinstance(labels[0], float):
        labels = labels.astype(int)  # TODO: LabelEncoder
    if str(labels.dtype).startswith('int'):
        labels = np.array(list('grbkcym'))[labels % 7]

    try:
        df = df.values
    except:
        pass

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[:, 0], df[:, 1], df[:, 2], zdir='z', s=20, c=labels, depthshade=depthshade)
    return fig


def offline_plotly(df=None):
    df = df or pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')
    print(df.head())

    data = []
    clusters = []
    colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 'rgb(77,175,74)']

    for i in range(len(df['Name'].unique())):
        name = df['Name'].unique()[i]
        color = colors[i]
        x = df[df['Name'] == name ]['SepalLength']
        y = df[df['Name'] == name ]['SepalWidth']
        z = df[df['Name'] == name ]['PetalLength']

        trace = dict(
            name = name,
            x = x, y = y, z = z,
            type = "scatter3d",    
            mode = 'markers',
            marker = dict( size=3, color=color, line=dict(width=0) ) )
        data.append( trace )

    layout = dict(
        width=800,
        height=550,
        autosize=False,
        title='Iris dataset',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'        
        ),
    )

    fig = dict(data=data, layout=layout)

    # IPython notebook
    # plotly.iplot(fig, filename='pandas-3d-iris', validate=False)

    url = plotly.offline.plot(fig, filename='pandas-3d-iris', validate=False)
