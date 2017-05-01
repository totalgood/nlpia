""" Explore the "topology" of High-D space and make anolgies in 2-3D

>>> from nlpia.plots import *
>>> df = pd.DataFrame(np.random.uniform(size=(1000,3)))
>>> labels = df.ix[:,:3].dot(np.array([3,-4,0])/5.) > 0
>>> df['c'] = labels.astype(int)
>>> scatter_3d(df, labels='c')
>>> plt.show()
"""
from seaborn import plt

import pandas as pd

from nlpia.plots import scatter_3d

np = pd.np


df = pd.DataFrame(pd.np.random.randn(1000, 10))

np = pd.np
plt.figure()
f = plt.figure()
ax = f.add_subplot(111)
axes = df.plot(kind='scatter', x=0, y=1, ax=ax)
plt.title('2D Normally Distributed')
plt.tight_layout()
plt.show()

scatter_3d(df)
plt.show()
df.values.dot(df.values)
np.do(df.values, df.values)
np.dot(df.values, df.values)
np.dot(df.values, df.values.T)
np.dot(df.values, df.values.T).shape
np.dot(df.values, df.values.T).shape.sum()
np.dot(df.values.T, df.values)
np.dot(df.values.T, df.values).shape
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,11)] for row in df])
row
for row in df:
    print(row)
    break
df
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,11)] for j, row in df.iterrows()])
norms
plt.hist(norms.values[:,9])
plt.hist(norms.values[:,8])
plt.show()
stds = []
for i in range(9):
    plt.hist(norms.values[:,i])
    plt.tight_layout()
    plt.show()
stds = []
for i in range(9):
    plt.hist(norms.values[:,i])
    plt.tight_layout()
pd.DataFrame(norms).describe()
hist
df
df = pd.DataFrame(np.random.uniform(1000,30))
df = pd.DataFrame(np.random.uniform(shape=(1000,30)))
df = pd.DataFrame(np.random.uniform((1000,30)))
df
df = pd.DataFrame(np.random.uniform(1000))
np.random.uniform(1000)
np.random.uniform?
df = pd.DataFrame(np.random.uniform(size=(1000,30)))


scatter_3d(df)
plt.show()
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,21)] for j, row in df.iterrows()])
stds = []
for i in range(19):
    plt.hist(norms.values[:,i])
    plt.tight_layout()
plt.show()
axes = pd.DataFrame(pd.np.random.uniform(size=(1000, 2)), columns=list('xy')).plot(kind='scatter', x='x', y='y', ax=ax)
axes.tight_layout()
plt.show()
axes.plot()
axes = pd.DataFrame(pd.np.random.uniform(size=(1000, 2)), columns=list('xy')).plot(kind='scatter', x='x', y='y', ax=ax)
plt.show()
f = plt.figure()
ax = f.add_subplot(111)
axes = pd.DataFrame(pd.np.random.uniform(size=(1000, 2)), columns=list('xy')).plot(kind='scatter', x='x', y='y', ax=ax)
plt.tight_layout()
plt.show()
df = pd.DataFrame(np.random.uniform(size=(1000,60)))
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,4,61)] for j, row in df.iterrows()])
stds = []
for i in range(15):
    plt.hist(norms.values[:,i])
    plt.tight_layout()
plt.show()
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,4,61)] for j, row in df.iterrows()])
norms.describe()
range(2,61,4)
list(range(2,61,4))
norms = pd.DataFrame([[np.linalg.norm(row[:i]) for i in range(2,61,4)] for j, row in df.iterrows()])
norms.describe()
stds = []
for i in range(15):
    plt.hist(norms.values[:,i])
    plt.tight_layout()
plt.show()
!git pull
hist

