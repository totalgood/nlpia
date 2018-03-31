import pandas as pd


def cleaner(row):
    cleaned = pd.np.array([s for i, s in enumerate(row.values)
                           if s not in ('Downtown', None) and (i > 3 or row[i + 1] != s)])
    if len(cleaned) == 2:
        cleaned = [cleaned[0], None, cleaned[1], None, None]
    else:
        cleaned = list(cleaned) + [None] * (5 - len(cleaned))
    if not pd.np.all(pd.np.array(row.values)[:3] == pd.np.array(cleaned)[:3]):
        print(row.values)
        print(cleaned)
    return list(cleaned)


def get_adwords(filename='/home/hobs/Downloads/AdWords API Location Criteria 2017-06-26.csv'):
    df = pd.read_csv(filename, header=0, index_col=0)
    df.columns = [c.replace(' ', '_').lower() for c in df.columns]
    canonical = pd.DataFrame([list(row) for row in df.canonical_name.str.split(',').values])
    canonical = canonical.dropna()
    cleancanon = canonical.apply(cleaner, axis=1)
    cleancanon.columns = 'city region country extra extra2'.split()
    return canonical
