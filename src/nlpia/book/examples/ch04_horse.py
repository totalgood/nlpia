"""
>>> import pandas as pd
>>> pd.set_option('display.max_columns', 6)  
>>> from sklearn.decomposition import PCA  
>>> import seaborn
>>> from matplotlib import pyplot as plt
>>> from nlpia.data.loaders import get_data

>> df = get_data('pointcloud').sample(1000)
>> pca = PCA(n_components=2)  
>> df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
>> df2d.plot(kind='scatter', x='x', y='y')
>> plt.show()
"""
import pandas as pd
pd.set_option('display.max_columns', 6)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # noqa
import seaborn
from matplotlib import pyplot as plt
from nlpia.data.loaders import get_data

df = get_data('pointcloud').sample(1000)
pca = PCA(n_components=2)  
df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y')
plt.show()
