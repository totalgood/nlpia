import pandas as pd
pd.read_csv('/home/hobs/Downloads/AdWords API Location Criteria 2017-06-26.csv', header=0, index=True)
pd.read_csv('/home/hobs/Downloads/AdWords API Location Criteria 2017-06-26.csv', header=0, index_col=0)
df = pd.read_csv('/home/hobs/Downloads/AdWords API Location Criteria 2017-06-26.csv', header=0, index_col=0)
df.columns
df.Name
df.columns
df.columns = [c.replace(' ', '_').lower() c for c in df.columns]
df.columns = [c.replace(' ', '_').lower() for c in df.columns]
df.columns
df.country_code
df.columns
df.canonical_name
canonical = df.canonical_name.apply(str.split?
canonical = pd.DataFrame(df.canonical_name.apply(str.split(sep=',')), columns='City Region Country'.split())
canonical = pd.DataFrame(df.canonical_name.str.split(sep=','), columns='City Region Country'.split())
canonical = pd.DataFrame(df.canonical_name.str.split(','), columns='City Region Country'.split())
canonical
df.canonical_name.str.split(',')
canonical = pd.DataFrame(df.canonical_name.str.split(',').values, columns='City Region Country'.split())
canonical = pd.DataFrame([list(row) for row in df.canonical_name.str.split(',').values], columns='City Region Country'.split())
canonical = pd.DataFrame([list(row) for row in df.canonical_name.str.split(',').values])
canonical
canonical[-1]
canonical.tail()
canonical
canonical.dropna()
def cleaner(row):
    return [s for i, s in enumerate(row.values) if s is not 'Downtown' and (i > 4 or row[i+1] != s)]
canonical.apply(cleaner, axis=1)
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s is not 'Downtown' and (i > 3 or row[i+1] != s)])
    if row.values != cleaned:
        print(row)
        print(cleaned)
    return cleaned
canonical.apply(cleaner, axis=1)
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if row.values != cleaned:
        print(row)
        print(cleaned)
    return cleaned
canonical.apply(cleaner, axis=1)
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if not (row.values == cleaned).all():
        print(row.values)
        print(cleaned)
    return cleaned
canonical.apply(cleaner, axis=1)
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if not pd.np.all(pd.np.array(row.values[:3]) == pd.np.array(cleaned[:3])):
        print(row.values)
        print(cleaned)
    return cleaned
canonical.apply(cleaner, axis=1)
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if not pd.np.all(pd.np.array(row.values)[:3] == pd.np.array(cleaned)[:3]):
        print(row.values)
        print(cleaned)
    return cleaned
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if not pd.np.all(pd.np.array(row.values)[:3] == pd.np.array(cleaned)[:3]):
        print(row.values)
        print(cleaned)
    if len(cleaned) == 2:
        cleaned = [cleaned[0], None, cleaned[1], None, None]
    else:
        cleaned = list(cleaned) + [None] * (5 - len(cleaned))
    return list(cleaned)
cleancanon = canonical.apply(cleaner, axis=1)
cleancanon
def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values) if s not in ('Downtown', None) and (i > 3 or row[i+1] != s)])
    if len(cleaned) == 2:
        cleaned = [cleaned[0], None, cleaned[1], None, None]
    else:
        cleaned = list(cleaned) + [None] * (5 - len(cleaned))
    if not pd.np.all(pd.np.array(row.values)[:3] == pd.np.array(cleaned)[:3]):
        print(row.values)
        print(cleaned)
    return list(cleaned)
cleancanon = canonical.apply(cleaner, axis=1)
newcanon
cleancanon
cleancanon.4
cleancanon[4]
cleancanon[4].notnull().sum()
cleancanon[3].notnull().sum()
cleancanon[3]
cleancanon[3][cleancanon[3].notnull()]
cleancanon[cleancanon[3].notnull()]
cleancanon.columns = 'City Region Country extra extra2'.split
cleancanon.columns = 'City Region Country extra extra2'.split()
cleancanon.columns = 'city region country extra extra2'.split()