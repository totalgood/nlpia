# Open Souce Contribution Ideas


### Basic

#### Pandas

- Add pd.DataFrame.__bool__ method that short circuits ValueError if self.empty

### Advanced

- make `DataFrame.from_csv` behave exactly the same as `pd.read_csv`
- add sql query ability
  - use pd.read_sql_query and sqlalchemy to dump the df(s) to an sqlite3 file, query, then return the results?
  - persist the "cache" of the dataframe(s) until python exit command is received or pandas shutdown or connection close or dataframe(s) deleted?

### SciPy SciKit Learn

- call `transform` and `tokenize` and other methods of Transforms and Models from within self.__iter__
