# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.graph_objs as go

import os

from seaborn import plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pandas as pd

import plotly.plotly as plotly
from plotly.offline.offline import _plot_html
from pugnlp.util import clean_columns
from plotly import graph_objs  # Scatter, Marker, Layout, YAxis, XAxis
import cufflinks as cf  # noqa

from nlpia.constants import DATA_PATH

np = pd.np

PLOTLY_HTML = """
<html>
<head>
<script type="text/javascript">
{plotlyjs}
</script>
</head>
<body>
{plotlyhtml}
</body>
</html>
"""

DEFAULT_PLOTLY_CONFIG = {
    'staticPlot': False,  # no interactivity, for export or image generation
    'workspace': False,  # we're in the workspace, so need toolbar etc
    'editable': False,  # we can edit titles, move annotations, etc
    'autosizable': False,  # plot will respect layout.autosize=true and infer its container size
    'fillFrame': False,  # if we DO autosize, do we fill the container or the screen?
    'scrollZoom': False,  # mousewheel or two-finger scroll zooms the plot
    'doubleClick': 'reset+autosize',  # double click interaction (false, 'reset', 'autosize' or 'reset+autosize')
    'showTips': True,  # new users see some hints about interactivity
    'showLink': True,  # link to open this plot in plotly
    'sendData': True,  # if we show a link, does it contain data or just link to a plotly file?
    'linkText': 'Edit chart',  # text appearing in the sendData link
    'displayModeBar': 'true',  # display the modebar (true, false, or 'hover')
    'displaylogo': False,  # add the plotly logo on the end of the modebar
    'plot3dPixelRatio': 2,  # increase the pixel ratio for 3D plot images
    'setBackground': 'opaque'  # fn to add the background color to a different container or 'opaque' to ensure there's white behind it
    }


def plotly_timeseries(df):
    fig = df.iplot([{
        'x': df.index,
        'y': df[col],
        'name': col
        } for col in df.columns], filename='cufflinks/simple-line')
    return fig


def scatter_3d(df, labels=None, depthshade=True):
    df = getattr(df, 'embedding_', df)
    labels = df[labels] if (isinstance(labels, (int, str, bytes)) and
                            labels in getattr(df, 'columns', set())) else labels
    labels = np.array(np.zeros(shape=(len(df),)) if labels is None else labels)
    try:
        labels = labels.astype(int)  # TODO: use LabelEncoder
    except (TypeError, AttributeError):
        pass
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


def get_array(df, x, default=None):
    if x is None:
        if default is None:
            x = df[df.columns[0]]
        else:
            x = df[default] if default in df else default
    elif isinstance(x, (pd.Series, pd.np.ndarray, list, tuple)):
        x = np.nd.array(x)
    else:
        x = df[x] if x in df.columns else df[df.columns[x]]
    return pd.np.array(x)


def offline_plotly_scatter3d(df, x=0, y=1, z=-1):
    """ Plot an offline scatter plot colored according to the categories in the 'name' column.

    >>> df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')
    >>> offline_plotly(df)
    """
    data = []
    # clusters = []
    colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 'rgb(77,175,74)']

    # df.columns = clean_columns(df.columns)

    x = get_array(df, x, default=0)
    y = get_array(df, y, default=1)
    z = get_array(df, z, default=-1)
    for i in range(len(df['name'].unique())):
        name = df['Name'].unique()[i]
        color = colors[i]
        x = x[pd.np.array(df['name'] == name)]
        y = y[pd.np.array(df['name'] == name)]
        z = z[pd.np.array(df['name'] == name)]

        trace = dict(
            name=name,
            x=x, y=y, z=z,
            type="scatter3d",
            mode='markers',
            marker=dict(size=3, color=color, line=dict(width=0)))
        data.append(trace)

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
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
    )

    fig = dict(data=data, layout=layout)

    # IPython notebook
    # plotly.iplot(fig, filename='pandas-3d-iris', validate=False)

    url = plotly.offline.plot(fig, filename='pandas-3d-iris', validate=False)
    return url


def annotate(row, ax, x='x', y='y', text='name', xytext=(7, -5), textcoords='offset points', **kwargs):
    """Add a text label to the plot of a DataFrame indicated by the provided axis (ax).

    Reference:
       https://stackoverflow.com/a/40979683/623735
    """
    # idx = row.name
    text = row[text] if text in row else str(text)
    x = row[x] if x in row else float(x)
    y = row[y] if y in row else float(y)
    ax.annotate(text, (row[x], row[y]), xytext=xytext, textcoords=textcoords, **kwargs)
    return row[text]


def offline_plotly_data(data, filename='plotly.html', config=None, validate=True,
                        default_width='100%', default_height=525, global_requirejs=False):
    """ Write a plotly scatter plot to HTML file that doesn't require server

    >>> import pandas as pd
    >>> from plotly.offline.offline import _plot_html
    >>> from plotly.graph_objs import Scatter, Marker, Layout, YAxis, XAxis
    >>> df = pd.read_csv('https://plot.ly/~etpinard/191.csv')
    >>> df.columns = [eval(c) if c[0] in '"\'' else str(c) for c in df.columns]
    >>> data = {'data': [
    >>>          Scatter(x=df[continent+', x'],
    >>>                  y=df[continent+', y'],
    >>>                  text=df[continent+', text'],
    >>>                  marker=Marker(size=df[continent+', size'], sizemode='area', sizeref=131868,),
    >>>                  mode='markers',
    >>>                  name=continent) for continent in ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']
    >>>      ],
    >>>      'layout': Layout(xaxis=XAxis(title='Life Expectancy'), yaxis=YAxis(title='GDP per Capita', type='log'))
    >>> }
    >>> html = offline_plotly_data(data)
    """
    config_default = dict(DEFAULT_PLOTLY_CONFIG)
    if config is not None:
        config_default.update(config)
    with open(os.path.join(DATA_PATH, 'plotly.js.min'), 'rt') as f:
        js = f.read()
    html, divid, width, height = _plot_html(
        data,
        config=config_default,
        validate=validate,
        default_width=default_width, default_height=default_height,
        global_requirejs=global_requirejs)
    html = PLOTLY_HTML.format(plotlyjs=js, plotlyhtml=html)
    if filename and isinstance(filename, str):
        with open(filename, 'wt') as f:
            f.write(html)
    return html


def normalize_etpinard_df(df='https://plot.ly/~etpinard/191.csv', columns='x y size text'.split(),
                          category_col='category', possible_categories=['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']):
    """Reformat a dataframe in etpinard's format for use in plot functions and sklearn models"""
    possible_categories = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'] if possible_categories is None else possible_categories
    df.columns = clean_columns(df.columns)
    df = pd.read_csv(df) if isinstance(df, str) else df
    columns = clean_columns(list(columns))
    df2 = pd.DataFrame(columns=columns)
    df2[category_col] = np.concatenate([np.array([categ] * len(df)) for categ in possible_categories])
    columns = zip(columns, [[clean_columns(categ + ', ' + column) for categ in possible_categories] for column in columns])
    for col, category_cols in columns:
        df2[col] = np.concatenate([df[label].values for label in category_cols])
    return df2


def offline_plotly_scatter_bubble(df, x='x', y='y', size_col='size', text_col='text',
                                  category_col='category', possible_categories=None,
                                  config={'displaylogo': False},
                                  layout={'type': 'log'},
                                  marker={'sizemin': 10, 'sizemode': 'area', 'sizeref': 1000},
                                  ):
    config_default = dict(DEFAULT_PLOTLY_CONFIG)
    marker_default = {
        'size': size_col,
        'sizemode': 'area',
        'sizeref': int(df[size_col].min() * .1)}
    marker_default.update(marker)
    size_col = marker_default.pop('size')
    layout_default = {
        'xaxis': graph_objs.XAxis(title=x),
        'yaxis': graph_objs.YAxis(title=y),
        'type': 'log',
        }
    layout_default.update(**layout)
    if config is not None:
        config_default.update(config)
    df.columns = clean_columns(df.columns)
    if possible_categories is None:
        if category_col in df.columns:
            category_labels = df[category_col]
        else:
            category_labels = np.array(category_col)
        possible_categories = list(category_labels.unique())
    if category_col in df:
        masks = [np.array(df[category_col] == label) for label in possible_categories]
    else:
        masks = [np.array([True] * len(df))] * len(possible_categories)
    print(marker_default)
    data = {'data': [
            graph_objs.Scatter(x=df[x][mask].values,
                               y=df[y][mask].values,
                               text=df[text_col][mask].values,
                               marker=graph_objs.Marker(size=df[size_col][mask] if size_col in df.columns else size_col,
                                                        **marker_default),
                               mode='markers',
                               name=str(category_name)) for (category_name, mask) in zip(possible_categories, masks)
            ],
            'layout': graph_objs.Layout(**layout_default)
            }
    return offline_plotly_data(data, config=config_default)
